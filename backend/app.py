from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
from cellpose import models
from PIL import Image, ImageEnhance
import io
import torch
import cv2
from ultralytics import YOLO
from scipy import ndimage
from skimage import measure, morphology, exposure
from skimage.segmentation import watershed
try:
    from skimage.feature import peak_local_maxima
except ImportError:
    # 在新版本的scikit-image中，peak_local_maxima可能在不同位置
    try:
        from scipy.ndimage import maximum_filter
        from scipy.ndimage import generate_binary_structure
        peak_local_maxima = None  # 如果需要使用，可以用scipy替代
    except ImportError:
        peak_local_maxima = None
from scipy.ndimage import distance_transform_edt
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 创建模型缓存目录
MODEL_CACHE_FOLDER = 'model_cache'
if not os.path.exists(MODEL_CACHE_FOLDER):
    os.makedirs(MODEL_CACHE_FOLDER)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 全局模型缓存字典
model_cache = {}

# 图像预处理函数
def enhance_image_contrast(img_array, contrast_factor=1.5, brightness_factor=1.1):
    """
    增强图像对比度和亮度，提升细胞边界清晰度
    内存优化：及时释放中间变量
    """
    # 转换为PIL图像进行增强
    if len(img_array.shape) == 3:
        img_pil = Image.fromarray(img_array.astype('uint8'))
    else:
        img_pil = Image.fromarray(img_array.astype('uint8'), mode='L')
    
    # 对比度增强
    enhancer = ImageEnhance.Contrast(img_pil)
    img_enhanced = enhancer.enhance(contrast_factor)
    del enhancer  # 释放内存
    
    # 亮度调整
    enhancer = ImageEnhance.Brightness(img_enhanced)
    img_final = enhancer.enhance(brightness_factor)
    del enhancer, img_enhanced  # 释放内存
    
    result = np.array(img_final)
    del img_pil, img_final  # 释放PIL图像内存
    return result

def calculate_optimal_size(original_shape, max_size=1024, min_size=512):
    """
    计算最优的缩放尺寸，保持长宽比
    """
    h, w = original_shape[:2]
    max_dim = max(h, w)
    
    if max_dim <= max_size:
        return h, w, 1.0  # 不需要缩放
    
    # 计算缩放比例
    scale_factor = max_size / max_dim
    new_h = max(int(h * scale_factor), min_size)
    new_w = max(int(w * scale_factor), min_size)
    
    return new_h, new_w, scale_factor

def resize_image_smart(img_array, target_size=None, max_size=1024):
    """
    智能缩放图像，保持长宽比
    内存优化：选择合适的插值算法和数据类型
    """
    original_shape = img_array.shape
    
    if target_size is None:
        new_h, new_w, scale_factor = calculate_optimal_size(original_shape, max_size)
    else:
        new_h, new_w = target_size
        scale_factor = min(new_h / original_shape[0], new_w / original_shape[1])
    
    # 如果不需要缩放，直接返回原图
    if scale_factor >= 0.99:
        return img_array.copy(), 1.0
    
    # 根据缩放比例选择合适的插值算法
    if scale_factor > 0.5:
        interpolation = cv2.INTER_LANCZOS4  # 高质量缩放
    else:
        interpolation = cv2.INTER_AREA  # 大幅缩放时使用AREA算法更快
    
    # 执行缩放
    resized = cv2.resize(img_array, (new_w, new_h), interpolation=interpolation)
    
    return resized, scale_factor

def map_masks_to_original(masks, scale_factor, original_shape):
    """
    将小图的识别结果映射回原图尺寸
    """
    if scale_factor == 1.0:
        return masks
    
    # 计算原图尺寸
    original_h, original_w = original_shape[:2]
    
    # 使用最近邻插值保持mask的整数标签
    mapped_masks = cv2.resize(
        masks.astype(np.float32), 
        (original_w, original_h), 
        interpolation=cv2.INTER_NEAREST
    ).astype(masks.dtype)
    
    return mapped_masks

# GPU detection and configuration
def detect_gpu_support():
    """
    检测系统是否支持GPU加速
    Returns:
        bool: True if GPU is available and supported, False otherwise
    """
    try:
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"GPU detected: {gpu_name} (CUDA devices: {gpu_count})")
            return True
        else:
            print("CUDA not available, using CPU")
            return False
    except Exception as e:
        print(f"Error detecting GPU: {e}, falling back to CPU")
        return False

# 全局GPU配置
USE_GPU = detect_gpu_support()

def get_cached_model(model_type):
    """
    获取缓存的Cellpose模型，如果不存在则创建并缓存
    支持Cellpose 3.0传统模型和Cellpose-SAM模型
    
    Args:
        model_type (str): 模型类型 (cyto, cyto2, nuclei, cellpose-sam等)
    
    Returns:
        models.CellposeModel: 缓存的Cellpose模型实例
    """
    if model_type not in model_cache:
        try:
            print(f"Loading model '{model_type}' for the first time...")
            # Map model names to pretrained_model names
            # Cellpose 3.0 traditional models
            model_map = {
                'cyto': 'cyto',
                'cyto2': 'cyto2', 
                'cyto3': 'cyto3',
                'nuclei': 'nuclei',
                # Cellpose-SAM models (using SAM backbone)
                'cellpose-sam': 'cpsam',
                'cpsam': 'cpsam'
            }
            pretrained_model = model_map.get(model_type, 'cyto2')  # default to cyto2
            
            # Create model with appropriate configuration
            if model_type in ['cellpose-sam', 'cpsam']:
                print(f"Loading Cellpose-SAM model with SAM backbone...")
                model_cache[model_type] = models.CellposeModel(gpu=USE_GPU, pretrained_model='cpsam')
            else:
                print(f"Loading Cellpose 3.0 traditional model...")
                model_cache[model_type] = models.CellposeModel(gpu=USE_GPU, pretrained_model=pretrained_model)
            
            device_info = "GPU" if USE_GPU else "CPU"
            print(f"Model '{model_type}' loaded on {device_info} and cached successfully (pretrained_model: {pretrained_model}).")
        except Exception as e:
            print(f"Error loading model {model_type}: {e}")
            return None
    else:
        print(f"Using cached model '{model_type}'.")
    
    return model_cache[model_type]

# YOLOv模型缓存
yolo_model_cache = {}

def extract_cell_contour_in_bbox(img_array, x1, y1, x2, y2):
    """
    在给定的边界框内提取细胞的实际轮廓形状
    使用图像处理技术而不是简单的矩形标注
    """
    try:
        # 确保输入图像是灰度图像
        if len(img_array.shape) == 3:
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = img_array.copy()
        
        # 确保坐标在有效范围内
        h, w = gray_img.shape[:2]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        
        # 提取边界框区域
        roi = gray_img[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            return None
        
        # 确保ROI是单通道8位图像
        if len(roi.shape) > 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        roi = roi.astype(np.uint8)
        
        # 创建与原图像相同大小的掩码
        full_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
        
        # 应用高斯模糊减少噪声，使用更大的核以获得更平滑的结果
        blurred = cv2.GaussianBlur(roi, (7, 7), 1.5)
        
        # 使用自适应阈值进行二值化
        # 调整参数以更好地适应细胞图像
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 3
        )
        
        # 形态学操作：优化参数以更好地处理细胞形状
        # 使用椭圆形核更适合细胞的圆形特征
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 先用小核去除噪声
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        # 再用大核填充空洞和连接断裂的边缘
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 选择最大的轮廓作为细胞轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 过滤过小的轮廓，降低阈值以保留更多有效轮廓
        contour_area = cv2.contourArea(largest_contour)
        roi_area = roi.shape[0] * roi.shape[1]
        if contour_area < roi_area * 0.05:  # 轮廓面积至少占ROI的5%
            return None
        
        # 使用轮廓近似来平滑轮廓边缘
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 在ROI上创建轮廓掩码
        roi_mask = np.zeros(roi.shape, dtype=np.uint8)
        cv2.fillPoly(roi_mask, [largest_contour], 255)
        
        # 将ROI掩码映射回原图像坐标
        full_mask[y1:y2, x1:x2] = roi_mask
        
        return full_mask
        
    except Exception as e:
        print(f"Error in extract_cell_contour_in_bbox: {str(e)}")
        return None

def get_cached_yolo_model(model_name='yolov8n.pt'):
    """
    获取缓存的YOLOv模型，如果不存在则创建并缓存
    
    Args:
        model_name (str): YOLOv模型名称
    
    Returns:
        YOLO: 缓存的YOLOv模型实例
    """
    if model_name not in yolo_model_cache:
        try:
            print(f"Loading YOLOv model '{model_name}' for the first time...")
            yolo_model_cache[model_name] = YOLO(model_name)
            device_info = "GPU" if USE_GPU else "CPU"
            print(f"YOLOv model '{model_name}' loaded on {device_info} and cached successfully.")
        except Exception as e:
            print(f"Error loading YOLOv model {model_name}: {e}")
            return None
    else:
        print(f"Using cached YOLOv model '{model_name}'.")
    
    return yolo_model_cache[model_name]

def analyze_with_yolo(img_array, confidence=0.25, iou_threshold=0.45, model_name='yolov8n.pt'):
    """
    使用YOLOv进行细胞检测和分割
    
    Args:
        img_array (np.ndarray): 输入图像数组
        confidence (float): 置信度阈值
        iou_threshold (float): IoU阈值
        model_name (str): YOLOv模型名称
    
    Returns:
        tuple: (masks, num_cells)
    """
    try:
        # 获取YOLOv模型
        model = get_cached_yolo_model(model_name)
        if model is None:
            raise Exception("Failed to load YOLOv model")
        
        # 确保图像是RGB格式（YOLOv需要3通道输入）
        if len(img_array.shape) == 2:  # 灰度图像
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:  # 单通道图像
            img_rgb = cv2.cvtColor(img_array.squeeze(), cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # 已经是RGB
            img_rgb = img_array
        else:
            raise Exception(f"Unsupported image format: {img_array.shape}")
        
        # 进行预测，增加NMS阈值来减少重叠检测
        results = model(img_rgb, conf=confidence, iou=0.5, verbose=False)
        
        # 创建分割掩码 - 使用原始图像尺寸
        original_shape = img_array.shape[:2]
        masks = np.zeros(original_shape, dtype=np.uint16)
        num_cells = 0
        
        print(f"YOLOv prediction results: {len(results)} result(s)")
        print(f"注意：YOLOv8预训练模型不是专门为细胞检测训练的，结果仅供参考")
        
        for result in results:
            print(f"Result boxes: {result.boxes.data.shape if result.boxes is not None else 'None'}")
            print(f"Result masks: {result.masks.data.shape if result.masks is not None else 'None'}")
            
            # 优先使用分割掩码，如果没有则使用检测框
            if result.masks is not None:
                print(f"Using segmentation masks: {len(result.masks.data)} masks")
                for i, mask in enumerate(result.masks.data):
                    # 将mask转换为numpy数组并调整大小到原始图像尺寸
                    mask_np = mask.cpu().numpy()
                    if mask_np.shape != original_shape:
                        mask_np = cv2.resize(mask_np, (original_shape[1], original_shape[0]))
                    
                    # 将mask添加到总掩码中，每个细胞使用不同的标签
                    cell_mask = (mask_np > 0.5).astype(np.uint16)
                    if np.sum(cell_mask) > 0:  # 确保mask不为空
                        num_cells += 1
                        masks[cell_mask > 0] = num_cells
            elif result.boxes is not None and len(result.boxes.data) > 0:
                print(f"Using detection boxes: {len(result.boxes.data)} boxes")
                # 限制检测数量，避免过多误检
                valid_boxes = 0
                boxes_data = result.boxes.data.cpu().numpy()
                accepted_boxes = []  # 存储已接受的框的中心点
                
                # 按置信度排序，优先处理高置信度的检测
                sorted_indices = np.argsort(boxes_data[:, 4])[::-1]
                
                for idx in sorted_indices:
                    if valid_boxes >= 8:  # 适当增加检测框数量到8个
                        break
                        
                    box = boxes_data[idx]
                    x1, y1, x2, y2, conf, cls = box
                    
                    # 使用适中的置信度阈值，平衡精度和召回率
                    if conf >= max(confidence, 0.7):  # 调整置信度阈值到0.7
                        # 计算框的面积，过滤过大或过小的框
                        box_area = (x2 - x1) * (y2 - y1)
                        img_area = original_shape[0] * original_shape[1]
                        area_ratio = box_area / img_area
                        
                        # 调整面积过滤范围（0.05% 到 2% 的图像面积）
                        if 0.0005 < area_ratio < 0.02:
                            # 检查长宽比，过滤过于细长的框
                            width = x2 - x1
                            height = y2 - y1
                            aspect_ratio = max(width, height) / min(width, height)
                            
                            if aspect_ratio < 2.0:  # 长宽比不超过2:1，允许更多形状变化
                                # 计算当前框的中心点
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                
                                # 检查与已接受框的距离，避免重叠
                                min_distance = min(original_shape) * 0.1  # 最小距离为图像较小边的10%
                                too_close = False
                                for accepted_center in accepted_boxes:
                                    distance = np.sqrt((center_x - accepted_center[0])**2 + (center_y - accepted_center[1])**2)
                                    if distance < min_distance:
                                        too_close = True
                                        break
                                
                                if not too_close:
                                    # 将坐标转换为整数并确保在图像范围内
                                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                                    x1 = max(0, min(x1, original_shape[1]-1))
                                    y1 = max(0, min(y1, original_shape[0]-1))
                                    x2 = max(0, min(x2, original_shape[1]-1))
                                    y2 = max(0, min(y2, original_shape[0]-1))
                                    
                                    if x2 > x1 and y2 > y1:  # 确保框有效
                                        # 在检测框内提取细胞的实际形状轮廓
                                        cell_mask = extract_cell_contour_in_bbox(img_array, x1, y1, x2, y2)
                                        
                                        if cell_mask is not None and np.sum(cell_mask) > 0:
                                            num_cells += 1
                                            valid_boxes += 1
                                            accepted_boxes.append((center_x, center_y))
                                            # 使用提取的细胞轮廓而不是矩形框
                                            masks[cell_mask > 0] = num_cells
                                            print(f"Added cell contour {num_cells}: bbox({x1},{y1}) to ({x2},{y2}), conf={conf:.3f}, class={int(cls)}, contour_pixels={np.sum(cell_mask)}")
                                        else:
                                            print(f"Failed to extract cell contour in bbox ({x1},{y1}) to ({x2},{y2}), skipping")
                
                print(f"Filtered to {valid_boxes} potential cells from {len(result.boxes.data)} detections (with distance filtering)")
            else:
                print("No boxes or masks found in result")
        
        print(f"YOLOv detected {num_cells} cells")
        return masks, num_cells
        
    except Exception as e:
        print(f"Error in YOLOv analysis: {str(e)}")
        raise e

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'filename': filename})

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    分析上传的细胞图像，支持Cellpose和YOLOv算法
    """
    # Check if file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Get algorithm type
        algorithm = request.form.get('algorithm', 'cellpose')  # Default to cellpose
        print(f"Selected algorithm: {algorithm}")
        
        # Read the file directly from memory without saving to disk
        file_content = file.read()
        img = Image.open(io.BytesIO(file_content))
        img_array = np.array(img)
        original_shape = img_array.shape
        
        # 获取图像预处理参数
        enable_preprocessing = request.form.get('enable_preprocessing', 'true').lower() == 'true'
        contrast_factor = float(request.form.get('contrast_factor', 1.5))
        brightness_factor = float(request.form.get('brightness_factor', 1.1))
        max_size = int(request.form.get('max_size', 1024))
        
        print(f"Original image shape: {original_shape}")
        print(f"Preprocessing enabled: {enable_preprocessing}")
        
        # 图像预处理 - 内存优化版本
        processed_img = img_array
        scale_factor = 1.0
        
        if enable_preprocessing:
            # 对比度和亮度增强
            enhanced_img = enhance_image_contrast(processed_img, contrast_factor, brightness_factor)
            print(f"Applied contrast enhancement: factor={contrast_factor}, brightness={brightness_factor}")
            
            # 智能缩放
            processed_img, scale_factor = resize_image_smart(enhanced_img, max_size=max_size)
            print(f"Resized image to: {processed_img.shape}, scale_factor: {scale_factor:.3f}")
            
            # 释放中间图像内存
            del enhanced_img
            import gc
            gc.collect()  # 强制垃圾回收
        
        if algorithm == 'yolo':
            # YOLOv parameters
            confidence = float(request.form.get('confidence', 0.25))
            iou_threshold = float(request.form.get('iou_threshold', 0.45))
            model_name = request.form.get('yolo_model', 'yolov8n.pt')
            
            # Analyze with YOLOv
            masks, num_cells = analyze_with_yolo(processed_img, confidence, iou_threshold, model_name)
            
            # 将结果映射回原图尺寸
            if enable_preprocessing and scale_factor != 1.0:
                masks = map_masks_to_original(masks, scale_factor, original_shape)
                print(f"Mapped YOLOv results back to original size: {masks.shape}")
            
            return jsonify({
                'num_cells': num_cells,
                'masks': masks.tolist(),
                'algorithm': 'yolo',
                'parameters_used': {
                    'algorithm': 'yolo',
                    'confidence': confidence,
                    'iou_threshold': iou_threshold,
                    'model_name': model_name,
                    'preprocessing': {
                        'enabled': enable_preprocessing,
                        'contrast_factor': contrast_factor,
                        'brightness_factor': brightness_factor,
                        'max_size': max_size,
                        'scale_factor': scale_factor
                    }
                }
            })
        
        elif algorithm in ['cellpose', 'cellpose-sam']:  # Cellpose algorithms
            # Determine model type based on algorithm
            if algorithm == 'cellpose-sam':
                model_type = request.form.get('model_type', 'cellpose-sam')  # Default to cellpose-sam
            else:
                model_type = request.form.get('model_type', 'cyto2')  # Default to cyto2 for Cellpose 3.0
            
            diameter = request.form.get('diameter', None)
            if diameter:
                diameter = float(diameter)
        
            # Parse channels parameter
            channels_str = request.form.get('channels', '[0,0]')
            try:
                channels = eval(channels_str)  # Parse string like "[0,0]" or "[1,2]"
            except:
                channels = [0, 0]  # Default grayscale
            
            # Additional parameters - 调整默认值以提高检测敏感度
            flow_threshold = float(request.form.get('flow_threshold', 0.4))
            cellprob_threshold = float(request.form.get('cellprob_threshold', -6.0))  # 降低阈值以提高敏感度

            # 使用缓存的Cellpose模型
            model = get_cached_model(model_type)
            # cellpose 4.0版本的eval方法返回值有变化
            # 优化CPU性能参数
            eval_params = {
                'diameter': diameter,
                'channels': channels,
                'flow_threshold': flow_threshold,
                'cellprob_threshold': cellprob_threshold,
                'normalize': True,  # 启用归一化以提高准确性
                'invert': False,    # 根据图像类型调整
            }
            
            # 如果使用CPU，添加性能优化参数
            if not USE_GPU:
                eval_params.update({
                    'batch_size': 8,    # 减小批处理大小以节省内存
                    'resample': True,   # 启用重采样
                })
            
            result = model.eval(processed_img, **eval_params)
            
            # 处理不同版本的返回值格式
            if len(result) == 4:
                masks, flows, styles, diams = result
            else:
                masks, flows, styles = result
                diams = None

            # 调试信息：打印masks的统计信息
            unique_values = np.unique(masks)
            print(f"Debug - Unique mask values: {unique_values}")
            print(f"Debug - Mask shape: {masks.shape}")
            print(f"Debug - Mask min/max: {masks.min()}/{masks.max()}")
            print(f"Debug - Parameters used: diameter={diameter}, flow_threshold={flow_threshold}, cellprob_threshold={cellprob_threshold}")
            
            # 将结果映射回原图尺寸
            if enable_preprocessing and scale_factor != 1.0:
                masks = map_masks_to_original(masks, scale_factor, original_shape)
                print(f"Mapped Cellpose results back to original size: {masks.shape}")
                # 重新计算unique values
                unique_values = np.unique(masks)
            
            # Calculate number of cells (exclude background)
            num_cells = len(unique_values) - 1
            print(f"Debug - Calculated num_cells: {num_cells}")
            
            # Return results with parameters used
            return jsonify({
                'num_cells': num_cells, 
                'masks': masks.tolist(),
                'algorithm': algorithm,
                'parameters_used': {
                    'algorithm': algorithm,
                    'model_type': model_type,
                    'diameter': diameter,
                    'channels': channels,
                    'flow_threshold': flow_threshold,
                    'cellprob_threshold': cellprob_threshold,
                    'preprocessing': {
                        'enabled': enable_preprocessing,
                        'contrast_factor': contrast_factor,
                        'brightness_factor': brightness_factor,
                        'max_size': max_size,
                        'scale_factor': scale_factor
                    }
                }
            })
        else:
            return jsonify({'error': f'Unsupported algorithm: {algorithm}. Supported algorithms: cellpose, cellpose-sam, yolo'}), 400
            
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/system-info', methods=['GET'])
def get_system_info():
    """
    获取系统信息，包括GPU支持状态和性能建议
    """
    try:
        system_info = {
            'gpu_available': USE_GPU,
            'device_type': 'GPU' if USE_GPU else 'CPU',
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if USE_GPU and torch.cuda.is_available():
            system_info.update({
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'performance_tip': 'GPU加速已启用，分析速度较快'
            })
        else:
            system_info.update({
                'performance_tip': 'CPU模式运行，分析速度较慢。如需提升性能，请安装CUDA支持的PyTorch版本'
            })
        
        return jsonify(system_info)
    except Exception as e:
        return jsonify({'error': f'Failed to get system info: {str(e)}'}), 500

def preload_common_models():
    """
    预加载常用的Cellpose模型以提高响应速度
    """
    common_models = ['cyto2', 'cyto', 'nuclei']
    print("Preloading common Cellpose models...")
    
    for model_type in common_models:
        try:
            get_cached_model(model_type)
        except Exception as e:
            print(f"Warning: Failed to preload model '{model_type}': {e}")
    
    print("Model preloading completed.")

if __name__ == '__main__':
    # 暂时跳过预加载，直接启动服务器
    # preload_common_models()
    
    print("Starting Flask server...")
    app.run(debug=True, port=5001)