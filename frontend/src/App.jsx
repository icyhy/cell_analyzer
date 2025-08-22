import React, { useState, useRef, useEffect } from 'react';
import * as GeoTIFF from 'geotiff';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [masks, setMasks] = useState([]);
  const [results, setResults] = useState('');
  const [imageObject, setImageObject] = useState(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  
  // 算法选择状态
  const [algorithm, setAlgorithm] = useState('cellpose');
  
  // Cellpose参数状态
  const [modelType, setModelType] = useState('cyto2');
  const [diameter, setDiameter] = useState('');
  const [channels, setChannels] = useState('[0,0]');
  const [flowThreshold, setFlowThreshold] = useState(0.4);
  const [cellprobThreshold, setCellprobThreshold] = useState(-6.0);
  
  // YOLOv参数状态
  const [confidence, setConfidence] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [yoloModel, setYoloModel] = useState('yolov8n.pt');
  
  // 图像预处理参数状态
  const [enablePreprocessing, setEnablePreprocessing] = useState(false);
  const [contrastFactor, setContrastFactor] = useState(1.2);
  const [brightnessFactor, setBrightnessFactor] = useState(1.0);
  const [maxSize, setMaxSize] = useState(1024);
  const [showPreprocessingPanel, setShowPreprocessingPanel] = useState(false);
  
  // 系统信息状态
  const [systemInfo, setSystemInfo] = useState(null);
  const [isLoadingSystemInfo, setIsLoadingSystemInfo] = useState(false);
  const [showSystemInfoModal, setShowSystemInfoModal] = useState(false);

  /**
   * 获取系统信息
   */
  const fetchSystemInfo = async () => {
    setIsLoadingSystemInfo(true);
    try {
      const response = await fetch('http://localhost:5001/system-info');
      if (response.ok) {
        const info = await response.json();
        setSystemInfo(info);
      } else {
        console.error('Failed to fetch system info');
      }
    } catch (error) {
      console.error('Error fetching system info:', error);
    } finally {
      setIsLoadingSystemInfo(false);
    }
  };

  /**
   * Handles the file input change event.
   * When a file is selected, it updates the state and clears previous results.
   * @param {React.ChangeEvent<HTMLInputElement>} e - The event object.
   */
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setMasks([]); // Clear old masks
    setResults(''); // Clear old results
    setImageObject(null); // Clear old image object
  };

  /**
   * Effect to fetch system info on component mount
   */
  useEffect(() => {
    fetchSystemInfo();
  }, []);

  /**
   * Effect to handle file processing.
   * When the file state changes, it attempts to read the TIFF file.
   */
  useEffect(() => {
    if (!file) {
      // Clear canvas if file is removed
      const canvas = canvasRef.current;
      if (canvas) {
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const processTiff = async () => {
      try {
        console.log("Processing file:", file);
        const tiff = await GeoTIFF.fromBlob(file);
        const image = await tiff.getImage();
        console.log("TIFF image loaded:", image);
        setImageObject(image);
      } catch (error) {
        console.error("Error reading TIFF file:", error);
        setResults(`Error reading TIFF file: ${error.message}`);
        setImageObject(null);
      }
    };

    processTiff();
  }, [file]);

  /**
   * Effect to draw the image and masks on the canvas.
   * This runs when the imageObject or masks state changes.
   */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageObject) return;

    const drawImage = async () => {
      try {
        const ctx = canvas.getContext('2d');
        const width = imageObject.getWidth();
        const height = imageObject.getHeight();
        canvas.width = width;
        canvas.height = height;

        const rasters = await imageObject.readRasters();
        const bitsPerSample = imageObject.getBitsPerSample()[0];
        const rgba = new Uint8ClampedArray(width * height * 4);

        if (rasters.length === 1) { // Grayscale
          const data = rasters[0];
          // Find actual min and max values in the data for better contrast
          let minVal = data[0];
          let maxVal = data[0];
          for (let i = 1; i < data.length; i++) {
            if (data[i] < minVal) minVal = data[i];
            if (data[i] > maxVal) maxVal = data[i];
          }
          
          // Avoid division by zero
          if (maxVal === minVal) {
            maxVal = minVal + 1;
          }
          
          console.log(`Grayscale image - Min: ${minVal}, Max: ${maxVal}, BitsPerSample: ${bitsPerSample}`);
          
          for (let i = 0; i < data.length; ++i) {
            // Normalize using actual min/max for better contrast
            const val = Math.round(((data[i] - minVal) / (maxVal - minVal)) * 255);
            rgba[i * 4] = val;
            rgba[i * 4 + 1] = val;
            rgba[i * 4 + 2] = val;
            rgba[i * 4 + 3] = 255;
          }
        } else if (rasters.length >= 3) { // RGB or RGBA
          const r = rasters[0];
          const g = rasters[1];
          const b = rasters[2];
          
          // Find actual min and max values for each channel
          let minR = r[0], maxR = r[0];
          let minG = g[0], maxG = g[0];
          let minB = b[0], maxB = b[0];
          
          for (let i = 1; i < r.length; i++) {
            if (r[i] < minR) minR = r[i];
            if (r[i] > maxR) maxR = r[i];
            if (g[i] < minG) minG = g[i];
            if (g[i] > maxG) maxG = g[i];
            if (b[i] < minB) minB = b[i];
            if (b[i] > maxB) maxB = b[i];
          }
          
          console.log(`RGB image - R: ${minR}-${maxR}, G: ${minG}-${maxG}, B: ${minB}-${maxB}`);
          
          for (let i = 0; i < r.length; ++i) {
            // Normalize each channel using its actual min/max
            rgba[i * 4] = Math.round(((r[i] - minR) / (maxR - minR || 1)) * 255);
            rgba[i * 4 + 1] = Math.round(((g[i] - minG) / (maxG - minG || 1)) * 255);
            rgba[i * 4 + 2] = Math.round(((b[i] - minB) / (maxB - minB || 1)) * 255);
            rgba[i * 4 + 3] = 255;
          }
          if (rasters.length === 4) {
            const a = rasters[3];
            let minA = a[0], maxA = a[0];
            for (let i = 1; i < a.length; i++) {
              if (a[i] < minA) minA = a[i];
              if (a[i] > maxA) maxA = a[i];
            }
            for (let i = 0; i < a.length; ++i) {
              rgba[i * 4 + 3] = Math.round(((a[i] - minA) / (maxA - minA || 1)) * 255);
            }
          }
        } else {
          const errorMessage = `Unsupported TIFF format with ${rasters.length} rasters.`;
          console.error(errorMessage);
          setResults(errorMessage);
          return;
        }

        const imageData = new ImageData(rgba, width, height);
        ctx.putImageData(imageData, 0, 0);

        // Draw masks if they exist
        if (masks && masks.length > 0) {
          ctx.fillStyle = 'rgba(255, 0, 255, 0.4)'; // Semi-transparent magenta
          for (let i = 0; i < masks.length; i++) {
            for (let j = 0; j < masks[i].length; j++) {
              if (masks[i][j] > 0) {
                ctx.fillRect(j, i, 1, 1);
              }
            }
          }
        }
      } catch (error) {
        console.error('Error drawing image:', error);
        setResults('Error drawing image. See console for details.');
      }
    };

    drawImage();
  }, [imageObject, masks]);

  /**
   * Handles the analyze button click event.
   * Sends the file to the backend for analysis.
   */
  /**
   * 处理分析按钮点击事件
   * 发送文件和参数到后端进行分析
   */
  const handleAnalyze = async () => {
    if (!file) {
      setResults('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('algorithm', algorithm);
    
    // 添加图像预处理参数
    formData.append('enable_preprocessing', enablePreprocessing.toString());
    formData.append('contrast_factor', contrastFactor.toString());
    formData.append('brightness_factor', brightnessFactor.toString());
    formData.append('max_size', maxSize.toString());
    
    if (algorithm === 'cellpose' || algorithm === 'cellpose-sam') {
      formData.append('model_type', modelType);
      if (diameter) {
        formData.append('diameter', diameter);
      }
      formData.append('channels', channels);
      formData.append('flow_threshold', flowThreshold.toString());
      formData.append('cellprob_threshold', cellprobThreshold.toString());
    } else if (algorithm === 'yolo') {
      formData.append('confidence', confidence.toString());
      formData.append('iou_threshold', iouThreshold.toString());
      formData.append('yolo_model', yoloModel);
    }

    try {
      setResults('Analyzing...');
      const response = await fetch('http://localhost:5001/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: 'Analysis failed with no specific error message.' }));
        throw new Error(errorData.message || 'Analysis failed');
      }

      const data = await response.json();
      const params = data.parameters_used;
      
      let resultText = `Cell Count: ${data.num_cells}\nAlgorithm: ${params.algorithm}`;
      
      // 显示预处理信息
      if (params.preprocessing) {
        const prep = params.preprocessing;
        resultText += `\n\n--- 图像预处理 ---\n启用: ${prep.enabled ? '是' : '否'}`;
        if (prep.enabled) {
          resultText += `\n对比度增强: ${prep.contrast_factor}\n亮度调整: ${prep.brightness_factor}\n最大尺寸: ${prep.max_size}px\n缩放比例: ${(prep.scale_factor * 100).toFixed(1)}%`;
        }
      }
      
      if (params.algorithm === 'cellpose' || params.algorithm === 'cellpose-sam') {
        resultText += `\n\n--- Cellpose参数 ---\nModel: ${params.model_type}\nDiameter: ${params.diameter || 'Auto'}\nChannels: ${JSON.stringify(params.channels)}\nFlow Threshold: ${params.flow_threshold}\nCellprob Threshold: ${params.cellprob_threshold}`;
      } else if (params.algorithm === 'yolo') {
        resultText += `\n\n--- YOLOv参数 ---\nModel: ${params.model_name}\nConfidence: ${params.confidence}\nIoU Threshold: ${params.iou_threshold}`;
      }
      
      setResults(resultText);
      setMasks(data.masks);
    } catch (error) {
      console.error('Error during analysis:', error);
      setResults(`Error during analysis: ${error.message}`);
    }
  };

  return (
    <div className="App-container">
      <div className="title-container">
        <h1 className="main-title">Cell Analyzer</h1>
        <button 
          className="info-icon-btn"
          onClick={() => setShowSystemInfoModal(true)}
          title="查看系统信息"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
          </svg>
        </button>
      </div>
      
      {/* 系统信息弹窗 */}
      {showSystemInfoModal && (
        <div className="modal-overlay" onClick={() => setShowSystemInfoModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>系统状态信息</h3>
              <button 
                className="modal-close-btn"
                onClick={() => setShowSystemInfoModal(false)}
              >
                ×
              </button>
            </div>
            <div className="modal-body">
              {isLoadingSystemInfo ? (
                <p>加载系统信息中...</p>
              ) : systemInfo ? (
                <div className="system-info">
                  <div className="info-item">
                    <span className="info-label">计算设备:</span>
                    <span className={`info-value ${systemInfo.device_type.toLowerCase()}`}>
                      {systemInfo.device_type}
                      {systemInfo.gpu_name && ` (${systemInfo.gpu_name})`}
                    </span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">PyTorch版本:</span>
                    <span className="info-value">{systemInfo.torch_version}</span>
                  </div>
                  <div className="info-item performance-tip">
                    <span className="info-label">性能提示:</span>
                    <span className="info-value">{systemInfo.performance_tip}</span>
                  </div>
                  <button 
                    onClick={fetchSystemInfo} 
                    className="refresh-btn"
                    disabled={isLoadingSystemInfo}
                  >
                    刷新状态
                  </button>
                </div>
              ) : (
                <p>无法获取系统信息</p>
              )}
            </div>
          </div>
        </div>
      )}
      <div className="App">
        <div className="file-management">
          <h3>File Management</h3>
          <div className="file-upload-container" onClick={() => fileInputRef.current.click()}>
            <input type="file" ref={fileInputRef} onChange={handleFileChange} accept=".tif,.tiff" style={{ display: 'none' }} />
            {file ? (
              <p className="file-name">{file.name}</p>
            ) : (
              <label className="file-upload-label">
                Click to select a .tif file
              </label>
            )}
          </div>
          <button onClick={handleAnalyze} className="btn btn-primary">Analyze</button>
          
          {/* 算法选择和参数控制界面 */}
          <div className="parameters-section">
            <h4>Analysis Parameters</h4>
            
            {/* 算法选择 */}
            <div className="parameter-group">
              <label htmlFor="algorithm">Algorithm:</label>
              <select 
                id="algorithm" 
                value={algorithm} 
                onChange={(e) => setAlgorithm(e.target.value)}
                className="parameter-input"
              >
                <option value="cellpose">Cellpose 3.0</option>
                <option value="cellpose-sam">Cellpose-SAM</option>
                <option value="yolo">YOLOv8</option>
              </select>
            </div>
            
            {/* 图像预处理参数 */}
            <div className="preprocessing-section">
              <label htmlFor="enable-preprocessing" className="preprocessing-label">
                <input 
                  id="enable-preprocessing"
                  type="checkbox" 
                  checked={enablePreprocessing} 
                  onChange={(e) => setEnablePreprocessing(e.target.checked)}
                  className="checkbox-input"
                />
                启用图像预处理
              </label>
              
              {enablePreprocessing && (
                <button 
                  type="button"
                  className="settings-btn"
                  onClick={() => setShowPreprocessingPanel(true)}
                  title="配置预处理参数"
                >
                  ⚙️
                </button>
              )}
            </div>
            
            {/* 预处理参数弹出面板 */}
            {showPreprocessingPanel && (
              <div className="modal-overlay" onClick={() => setShowPreprocessingPanel(false)}>
                <div className="preprocessing-panel" onClick={(e) => e.stopPropagation()}>
                  <div className="panel-header">
                    <h4>图像预处理参数设置</h4>
                    <button 
                      className="close-btn"
                      onClick={() => setShowPreprocessingPanel(false)}
                    >
                      ✕
                    </button>
                  </div>
                  
                  <div className="panel-content">
                    <div className="parameter-group">
                      <label htmlFor="contrast-factor">对比度增强 (1.0-3.0):</label>
                      <input 
                        id="contrast-factor"
                        type="number" 
                        value={contrastFactor} 
                        onChange={(e) => setContrastFactor(parseFloat(e.target.value))}
                        className="parameter-input"
                        min="1.0"
                        max="3.0"
                        step="0.1"
                      />
                      <small>提升细胞边界清晰度</small>
                    </div>
                    
                    <div className="parameter-group">
                      <label htmlFor="brightness-factor">亮度调整 (0.5-2.0):</label>
                      <input 
                        id="brightness-factor"
                        type="number" 
                        value={brightnessFactor} 
                        onChange={(e) => setBrightnessFactor(parseFloat(e.target.value))}
                        className="parameter-input"
                        min="0.5"
                        max="2.0"
                        step="0.1"
                      />
                      <small>调整图像整体亮度</small>
                    </div>
                    
                    <div className="parameter-group">
                      <label htmlFor="max-size">最大处理尺寸 (像素):</label>
                      <select 
                        id="max-size" 
                        value={maxSize} 
                        onChange={(e) => setMaxSize(parseInt(e.target.value))}
                        className="parameter-input"
                      >
                        <option value="512">512px (最快)</option>
                        <option value="768">768px (平衡)</option>
                        <option value="1024">1024px (推荐)</option>
                        <option value="1536">1536px (高质量)</option>
                        <option value="2048">2048px (最高质量)</option>
                      </select>
                      <small>大图像将缩放到此尺寸进行处理</small>
                    </div>
                    
                    <div className="panel-actions">
                      <button 
                        type="button"
                        className="reset-btn"
                        onClick={() => {
                          setContrastFactor(1.2);
                          setBrightnessFactor(1.0);
                          setMaxSize(1024);
                        }}
                      >
                        重置默认值
                      </button>
                      <button 
                        type="button"
                        className="confirm-btn"
                        onClick={() => setShowPreprocessingPanel(false)}
                      >
                        确定
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Cellpose 3.0参数 */}
            {algorithm === 'cellpose' && (
              <>
                <div className="parameter-group">
                  <label htmlFor="model-type">Model Type:</label>
                  <select 
                    id="model-type" 
                    value={modelType} 
                    onChange={(e) => setModelType(e.target.value)}
                    className="parameter-input"
                  >
                    <option value="cyto">Cyto (General cells)</option>
                    <option value="cyto2">Cyto2 (Improved general)</option>
                    <option value="nuclei">Nuclei (Cell nuclei)</option>
                    <option value="cyto3">Cyto3 (Latest general)</option>
                  </select>
                </div>
            
                <div className="parameter-group">
                  <label htmlFor="diameter">Cell Diameter (pixels, leave empty for auto):</label>
                  <input 
                    id="diameter"
                    type="number" 
                    value={diameter} 
                    onChange={(e) => setDiameter(e.target.value)}
                    placeholder="Auto detect"
                    className="parameter-input"
                    min="1"
                  />
                </div>
                
                <div className="parameter-group">
                  <label htmlFor="channels">Channels [cytoplasm, nucleus]:</label>
                  <select 
                    id="channels" 
                    value={channels} 
                    onChange={(e) => setChannels(e.target.value)}
                    className="parameter-input"
                  >
                    <option value="[0,0]">Grayscale [0,0]</option>
                    <option value="[1,0]">Red cytoplasm [1,0]</option>
                    <option value="[2,0]">Green cytoplasm [2,0]</option>
                    <option value="[0,1]">Blue nucleus [0,1]</option>
                    <option value="[1,2]">Red cyto, Green nucleus [1,2]</option>
                    <option value="[2,1]">Green cyto, Red nucleus [2,1]</option>
                  </select>
                </div>
                
                <div className="parameter-group">
                  <label htmlFor="flow-threshold">Flow Threshold (0.0-3.0):</label>
                  <input 
                    id="flow-threshold"
                    type="number" 
                    value={flowThreshold} 
                    onChange={(e) => setFlowThreshold(parseFloat(e.target.value))}
                    className="parameter-input"
                    min="0"
                    max="3"
                    step="0.1"
                  />
                </div>
                
                <div className="parameter-group">
                  <label htmlFor="cellprob-threshold">Cell Probability Threshold (-6.0 to 6.0):</label>
                  <input 
                    id="cellprob-threshold"
                    type="number" 
                    value={cellprobThreshold} 
                    onChange={(e) => setCellprobThreshold(parseFloat(e.target.value))}
                    className="parameter-input"
                    min="-6"
                    max="6"
                    step="0.1"
                  />
                </div>
              </>
            )}
            
            {/* Cellpose-SAM参数 */}
            {algorithm === 'cellpose-sam' && (
              <>
                <div className="parameter-group">
                  <label htmlFor="model-type">Model Type:</label>
                  <select 
                    id="model-type" 
                    value={modelType} 
                    onChange={(e) => setModelType(e.target.value)}
                    className="parameter-input"
                  >
                    <option value="cellpose-sam">Cellpose-SAM (SAM backbone)</option>
                  </select>
                </div>
            
                <div className="parameter-group">
                  <label htmlFor="diameter">Cell Diameter (pixels, leave empty for auto):</label>
                  <input 
                    id="diameter"
                    type="number" 
                    value={diameter} 
                    onChange={(e) => setDiameter(e.target.value)}
                    placeholder="Auto detect"
                    className="parameter-input"
                    min="1"
                  />
                </div>
                
                <div className="parameter-group">
                  <label htmlFor="channels">Channels [cytoplasm, nucleus]:</label>
                  <select 
                    id="channels" 
                    value={channels} 
                    onChange={(e) => setChannels(e.target.value)}
                    className="parameter-input"
                  >
                    <option value="[0,0]">Grayscale [0,0]</option>
                    <option value="[1,0]">Red cytoplasm [1,0]</option>
                    <option value="[2,0]">Green cytoplasm [2,0]</option>
                    <option value="[0,1]">Blue nucleus [0,1]</option>
                    <option value="[1,2]">Red cyto, Green nucleus [1,2]</option>
                    <option value="[2,1]">Green cyto, Red nucleus [2,1]</option>
                  </select>
                </div>
                
                <div className="parameter-group">
                  <label htmlFor="flow-threshold">Flow Threshold (0.0-3.0):</label>
                  <input 
                    id="flow-threshold"
                    type="number" 
                    value={flowThreshold} 
                    onChange={(e) => setFlowThreshold(parseFloat(e.target.value))}
                    className="parameter-input"
                    min="0"
                    max="3"
                    step="0.1"
                  />
                </div>
                
                <div className="parameter-group">
                  <label htmlFor="cellprob-threshold">Cell Probability Threshold (-6.0 to 6.0):</label>
                  <input 
                    id="cellprob-threshold"
                    type="number" 
                    value={cellprobThreshold} 
                    onChange={(e) => setCellprobThreshold(parseFloat(e.target.value))}
                    className="parameter-input"
                    min="-6"
                    max="6"
                    step="0.1"
                  />
                </div>
              </>
            )}
            
            {/* YOLOv参数 */}
            {algorithm === 'yolo' && (
              <>
                <div className="parameter-group">
                  <label htmlFor="yolo-model">YOLOv Model:</label>
                  <select 
                    id="yolo-model" 
                    value={yoloModel} 
                    onChange={(e) => setYoloModel(e.target.value)}
                    className="parameter-input"
                  >
                    <option value="yolov8n.pt">YOLOv8n (Nano - Fast)</option>
                    <option value="yolov8s.pt">YOLOv8s (Small)</option>
                    <option value="yolov8m.pt">YOLOv8m (Medium)</option>
                    <option value="yolov8l.pt">YOLOv8l (Large)</option>
                    <option value="yolov8x.pt">YOLOv8x (Extra Large)</option>
                  </select>
                </div>
                
                <div className="parameter-group">
                  <label htmlFor="confidence">Confidence Threshold (0.0-1.0):</label>
                  <input 
                    id="confidence"
                    type="number" 
                    value={confidence} 
                    onChange={(e) => setConfidence(parseFloat(e.target.value))}
                    className="parameter-input"
                    min="0"
                    max="1"
                    step="0.01"
                  />
                </div>
                
                <div className="parameter-group">
                  <label htmlFor="iou-threshold">IoU Threshold (0.0-1.0):</label>
                  <input 
                    id="iou-threshold"
                    type="number" 
                    value={iouThreshold} 
                    onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
                    className="parameter-input"
                    min="0"
                    max="1"
                    step="0.01"
                  />
                </div>
              </>
            )}
          </div>
        </div>
        <div className="image-canvas">
          <canvas ref={canvasRef}></canvas>
        </div>
        <div className="analysis-results">
          <h3>Analysis Results</h3>
          <div className="results-container">
            <pre>{results}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
