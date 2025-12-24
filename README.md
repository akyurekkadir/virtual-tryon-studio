# 👔 Virtual Try-On Studio

An AI-powered virtual try-on application that uses ComfyUI backend with IDM-VTON for realistic garment visualization.

![Virtual Try-On Studio](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **Virtual Try-On**: Realistic garment fitting using IDM-VTON AI model
- **Color Harmony Analysis**: Automatic color matching score using K-Means clustering
- **Real-time Progress**: WebSocket-based live progress tracking
- **Debug Mode**: Visual mask preview for troubleshooting
- **Modern UI**: Clean, minimalist interface with maximum readability
- **Multiple Output Support**: Tries multiple methods to retrieve results (API, direct file access, newest file)

## 🎯 Demo

Upload a model photo and a garment photo, select category, and get:
- Virtual try-on result
- Color harmony score (0-100)
- Generated mask (debug view)

## 🛠️ Requirements

- Python 3.8+
- ComfyUI with IDM-VTON nodes installed
- Required models:
  - SAM (Segment Anything Model)
  - GroundingDINO
  - DensePose
  - IDM-VTON pipeline

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/virtual-tryon-studio.git
cd virtual-tryon-studio
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure paths

Edit `app.py` and update these paths:

```python
COMFYUI_INPUT_PATH = "/path/to/ComfyUI/input"
COMFYUI_OUTPUT_PATH = "/path/to/ComfyUI/output"
COMFYUI_TEMP_PATH = "/path/to/ComfyUI/temp"
```

### 4. Start ComfyUI

```bash
cd /path/to/ComfyUI
python main.py
```

### 5. Run the application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

## 🎨 Usage

1. **Upload Images**
   - Model photo (person wearing simple clothes)
   - Garment photo (clothing item on plain background)

2. **Configure**
   - Select category: Upper Body or Lower Body
   - Enter garment description (e.g., "Red polo shirt")

3. **Try On**
   - Click "Try On" button
   - Wait ~15-20 minutes for processing
   - View real-time progress

4. **Results**
   - Virtual try-on image
   - Color harmony score
   - Download result

## 🧪 Test Color Harmony

Use the sidebar test feature to quickly analyze any image's color harmony without running the full workflow.

## 🔧 Technical Details

### Workflow Pipeline

1. **Image Loading**: Load model and garment images
2. **Preprocessing**: Resize images to 624x880
3. **Segmentation**: Use GroundingDINO + SAM for mask generation
4. **Pose Detection**: Extract pose information with DensePose
5. **Try-On**: Apply IDM-VTON with detected pose and mask
6. **Color Analysis**: K-Means clustering for dominant color extraction

### Color Harmony Algorithm

- Splits image horizontally (upper/lower body)
- Extracts dominant colors using K-Means (n=5)
- Converts to HSV color space
- Calculates harmony score based on:
  - Complementary colors (150°+ hue diff): 88-98 points
  - Analogous colors (<30° hue diff): 80-92 points
  - Neutral colors (low saturation): 85-96 points
  - Other combinations: 55-85 points

### Debug Mode

Currently hardcoded for testing:
- Mask prompt: "shirt"
- Threshold: 0.3

To disable debug mode, edit `update_workflow()` function.

## 📊 Architecture

```
User Input → Streamlit UI → ComfyUI API → Workflow Execution
                ↓                              ↓
         File Upload                   WebSocket Progress
                ↓                              ↓
         Color Analysis ← Output Retrieval ← Result
```

## 🚀 Performance

- Processing time: ~15-20 minutes (depends on hardware)
- GPU recommended (CUDA support)
- RAM: Minimum 8GB, recommended 16GB+

## 🐛 Troubleshooting

### "Output not found" error
- Check ComfyUI is running (`http://127.0.0.1:8188`)
- Verify output/temp folder paths
- Check disk space
- View debug mask to verify mask generation

### Black mask issue
- Threshold is set to 0.3 (debug mode)
- Try adjusting threshold in workflow template
- Check model is properly visible in input image

### Slow processing
- Ensure GPU is being used
- Close other GPU-intensive applications
- Check ComfyUI logs for errors

## 📝 Configuration

### Workflow Customization

Edit `WORKFLOW_TEMPLATE` in `app.py`:
- Adjust image size (default: 624x880)
- Change inference steps (default: 20)
- Modify guidance scale (default: 2.0)
- Update seed for reproducibility

### UI Customization

Edit `CUSTOM_CSS` in `app.py` for styling changes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Powerful node-based UI
- [IDM-VTON](https://github.com/yisol/IDM-VTON) - Virtual try-on model
- [SAM](https://github.com/facebookresearch/segment-anything) - Segmentation model
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - Object detection

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ using Streamlit and ComfyUI**

