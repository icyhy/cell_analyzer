from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
from cellpose import models
from PIL import Image
import io

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

def get_cached_model(model_type):
    """
    获取缓存的Cellpose模型，如果不存在则创建并缓存
    
    Args:
        model_type (str): 模型类型 (cyto, cyto2, nuclei, cyto3等)
    
    Returns:
        models.Cellpose: 缓存的Cellpose模型实例
    """
    if model_type not in model_cache:
        print(f"Loading model '{model_type}' for the first time...")
        # 创建模型实例，Cellpose会自动处理模型文件的下载和缓存
        model_cache[model_type] = models.Cellpose(gpu=False, model_type=model_type)
        print(f"Model '{model_type}' loaded and cached successfully.")
    else:
        print(f"Using cached model '{model_type}'.")
    
    return model_cache[model_type]

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
    分析上传的细胞图像，支持多种Cellpose模型和参数配置
    """
    # Check if file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Get parameters from form data with defaults
        model_type = request.form.get('model_type', 'cyto2')  # Default to cyto2 for better performance
        diameter = request.form.get('diameter', None)
        if diameter:
            diameter = float(diameter)
        
        # Parse channels parameter
        channels_str = request.form.get('channels', '[0,0]')
        try:
            channels = eval(channels_str)  # Parse string like "[0,0]" or "[1,2]"
        except:
            channels = [0, 0]  # Default grayscale
        
        # Additional parameters
        flow_threshold = float(request.form.get('flow_threshold', 0.4))
        cellprob_threshold = float(request.form.get('cellprob_threshold', 0.0))
        
        # Read the file directly from memory without saving to disk
        file_content = file.read()
        img = Image.open(io.BytesIO(file_content))
        img_array = np.array(img)

        # 使用缓存的Cellpose模型
        model = get_cached_model(model_type)
        masks, flows, styles, diams = model.eval(
            img_array, 
            diameter=diameter, 
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )

        # Calculate number of cells (exclude background)
        num_cells = len(np.unique(masks)) - 1
        
        # Return results with parameters used
        return jsonify({
            'num_cells': num_cells, 
            'masks': masks.tolist(),
            'parameters_used': {
                'model_type': model_type,
                'diameter': diameter,
                'channels': channels,
                'flow_threshold': flow_threshold,
                'cellprob_threshold': cellprob_threshold
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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
    # 预加载常用模型
    preload_common_models()
    
    print("Starting Flask server...")
    app.run(debug=True, port=5001)