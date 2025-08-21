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
  
  // Cellpose参数状态
  const [modelType, setModelType] = useState('cyto2');
  const [diameter, setDiameter] = useState('');
  const [channels, setChannels] = useState('[0,0]');
  const [flowThreshold, setFlowThreshold] = useState(0.4);
  const [cellprobThreshold, setCellprobThreshold] = useState(0.0);

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
    formData.append('model_type', modelType);
    if (diameter) {
      formData.append('diameter', diameter);
    }
    formData.append('channels', channels);
    formData.append('flow_threshold', flowThreshold.toString());
    formData.append('cellprob_threshold', cellprobThreshold.toString());

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
      setResults(`Cell Count: ${data.num_cells}\nModel: ${params.model_type}\nDiameter: ${params.diameter || 'Auto'}\nChannels: ${JSON.stringify(params.channels)}`);
      setMasks(data.masks);
    } catch (error) {
      console.error('Error during analysis:', error);
      setResults(`Error during analysis: ${error.message}`);
    }
  };

  return (
    <div className="App-container">
      <h1 className="main-title">Cell Analyzer</h1>
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
          
          {/* Cellpose参数控制界面 */}
          <div className="parameters-section">
            <h4>Cellpose Parameters</h4>
            
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
