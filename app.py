import streamlit as st
import requests
import json
import uuid
import os
import time
from pathlib import Path
from PIL import Image
import io
import websocket
import threading
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
import colorsys

# ============================================================================
# CONFIGURATION
# ============================================================================
COMFYUI_SERVER = "http://127.0.0.1:8188"
COMFYUI_INPUT_PATH = "/Applications/Data/Packages/ComfyUI/input"
COMFYUI_OUTPUT_PATH = "/Applications/Data/Packages/ComfyUI/output"
COMFYUI_TEMP_PATH = "/Applications/Data/Packages/ComfyUI/temp"

# ============================================================================
# CUSTOM CSS - ULTRA CLEAN & MODERN
# ============================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Reset and base */
    .main {
        background: #ffffff;
        padding: 2rem 4rem;
    }
    
    .stApp {
        background: #ffffff;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 3rem;
        max-width: 1400px;
    }
    
    /* Headers - LARGE & READABLE */
    h1 {
        color: #000000 !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.2 !important;
    }
    
    h2 {
        color: #000000 !important;
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        margin: 2.5rem 0 1.5rem 0 !important;
        letter-spacing: -0.01em !important;
    }
    
    h3 {
        color: #000000 !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        margin: 1.5rem 0 1rem 0 !important;
    }
    
    /* Paragraphs and text - DARK & CLEAR */
    p, div, span, label {
        color: #000000 !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    
    /* Subtitle */
    .subtitle {
        color: #666666 !important;
        font-size: 1.25rem !important;
        margin-bottom: 3rem !important;
    }
    
    /* Section divider */
    .section-divider {
        height: 2px;
        background: #000000;
        margin: 3rem 0;
        opacity: 0.1;
    }
    
    /* File uploader - CLEAN & SOLID */
    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 3px solid #000000;
        border-radius: 4px;
        padding: 2rem;
    }
    
    div[data-testid="stFileUploader"] label {
        color: #000000 !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stFileUploader"] button {
        background: #000000 !important;
        color: #ffffff !important;
        border: 3px solid #000000 !important;
        border-radius: 4px !important;
        padding: 1rem 2rem !important;
        font-weight: 900 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.05em !important;
    }
    
    div[data-testid="stFileUploader"] section {
        background: #f8f8f8 !important;
        border: 2px solid #000000 !important;
        border-radius: 4px !important;
        padding: 2rem !important;
    }
    
    div[data-testid="stFileUploader"] section > div {
        color: #000000 !important;
    }
    
    /* Keep icon and button, hide only the long text */
    div[data-testid="stFileUploader"] small {
        display: none !important;
    }
    
    div[data-testid="stFileUploader"] section {
        min-height: 120px !important;
    }
    
    /* Text input - READABLE */
    .stTextInput label {
        color: #000000 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.75rem !important;
    }
    
    .stTextInput input {
        border: 2px solid #000000 !important;
        border-radius: 4px !important;
        padding: 1rem !important;
        font-size: 1.1rem !important;
        color: #000000 !important;
        background: #ffffff !important;
    }
    
    .stTextInput input:focus {
        border-color: #000000 !important;
        box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Radio buttons - BOLD */
    .stRadio label {
        color: #000000 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    .stRadio > div {
        background: #f8f8f8;
        border: 2px solid #000000;
        border-radius: 4px;
        padding: 1.5rem;
    }
    
    .stRadio > div > label {
        font-size: 1.05rem !important;
        padding: 0.5rem 0 !important;
    }
    
    /* Buttons - MAXIMUM CONTRAST */
    .stButton > button {
        background: #000000 !important;
        color: #ffffff !important;
        border: 3px solid #000000 !important;
        border-radius: 4px !important;
        padding: 1.25rem 3rem !important;
        font-size: 1.3rem !important;
        font-weight: 900 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        transition: all 0.2s ease !important;
        text-shadow: none !important;
    }
    
    .stButton > button:hover {
        background: #333333 !important;
        border-color: #333333 !important;
        transform: translateY(-1px);
    }
    
    .stButton > button p {
        color: #ffffff !important;
        font-size: 1.3rem !important;
        font-weight: 900 !important;
    }
    
    .stDownloadButton > button {
        background: #000000 !important;
        color: #ffffff !important;
        border: 3px solid #000000 !important;
        border-radius: 4px !important;
        padding: 1.25rem 2rem !important;
        font-size: 1.2rem !important;
        font-weight: 900 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }
    
    .stDownloadButton > button:hover {
        background: #333333 !important;
        border-color: #333333 !important;
    }
    
    .stDownloadButton > button p {
        color: #ffffff !important;
        font-size: 1.2rem !important;
        font-weight: 900 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: #000000 !important;
        height: 8px !important;
        border-radius: 4px !important;
    }
    
    .stProgress > div > div {
        background: #f0f0f0 !important;
        border-radius: 4px !important;
    }
    
    /* Images - CLEAN BORDERS */
    .stImage {
        border: 2px solid #000000;
        border-radius: 4px;
    }
    
    /* Sidebar - CLEAN */
    section[data-testid="stSidebar"] {
        background: #f8f8f8;
        border-right: 2px solid #000000;
        padding: 2rem 1.5rem;
    }
    
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {
        color: #000000 !important;
        font-size: 0.95rem !important;
    }
    
    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        padding: 1rem !important;
        font-size: 1.2rem !important;
        font-weight: 900 !important;
        letter-spacing: 0.08em !important;
        background: #000000 !important;
        color: #ffffff !important;
        border: 3px solid #000000 !important;
    }
    
    section[data-testid="stSidebar"] .stButton button p {
        color: #ffffff !important;
        font-size: 1.2rem !important;
        font-weight: 900 !important;
    }
    
    /* Alert boxes - HIGH CONTRAST */
    .stAlert {
        background: #f8f8f8 !important;
        border: 3px solid #000000 !important;
        border-radius: 4px !important;
        color: #000000 !important;
        padding: 1rem 1.5rem !important;
    }
    
    .stAlert p,
    .stAlert div,
    .stAlert span {
        color: #000000 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    .stSuccess {
        background: #e6ffe6 !important;
        border-color: #000000 !important;
    }
    
    .stError {
        background: #ffe6e6 !important;
        border-color: #000000 !important;
    }
    
    .stInfo {
        background: #e6f3ff !important;
        border-color: #000000 !important;
    }
    
    .stSpinner > div {
        color: #000000 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    /* Score card - BOLD & CLEAR */
    .score-container {
        background: #000000;
        color: #ffffff;
        padding: 2rem 1.5rem;
        border-radius: 4px;
        text-align: center;
        margin: 1.5rem 0;
        max-width: 400px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .score-number {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.75rem;
        letter-spacing: -0.02em;
    }
    
    .score-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff !important;
    }
    
    /* Test section */
    .test-container {
        background: #f8f8f8;
        border: 3px solid #000000;
        border-radius: 4px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .test-container div {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #000000 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Markdown text */
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span {
        color: #000000 !important;
    }
    
    /* Status text */
    .stText {
        color: #000000 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    /* Remove extra padding */
    .element-container {
        margin-bottom: 0.5rem;
    }
</style>
"""

# ============================================================================
# WORKFLOW TEMPLATE
# ============================================================================
WORKFLOW_TEMPLATE = {
    "12": {
        "inputs": {"weight_dtype": "float16"},
        "class_type": "PipelineLoader"
    },
    "14": {
        "inputs": {"image": "human_image_placeholder.png"},
        "class_type": "LoadImage"
    },
    "15": {
        "inputs": {"image": "garment_image_placeholder.png"},
        "class_type": "LoadImage"
    },
    "20": {
        "inputs": {"images": ["35", 0]},
        "class_type": "PreviewImage"
    },
    "27": {
        "inputs": {"model_name": "sam_vit_b (375MB)"},
        "class_type": "SAMModelLoader (segment anything)"
    },
    "28": {
        "inputs": {"model_name": "GroundingDINO_SwinT_OGC (694MB)"},
        "class_type": "GroundingDinoModelLoader (segment anything)"
    },
    "29": {
        "inputs": {
            "prompt": "shirt",
            "threshold": 0.3,
            "sam_model": ["27", 0],
            "grounding_dino_model": ["28", 0],
            "image": ["50", 0]
        },
        "class_type": "GroundingDinoSAMSegment (segment anything)"
    },
    "31": {
        "inputs": {"mask": ["29", 1]},
        "class_type": "MaskToImage"
    },
    "32": {
        "inputs": {"images": ["31", 0]},
        "class_type": "PreviewImage"
    },
    "33": {
        "inputs": {
            "model": "densepose_r50_fpn_dl.torchscript",
            "cmap": "Parula (CivitAI)",
            "resolution": 768,
            "image": ["50", 0]
        },
        "class_type": "DensePosePreprocessor"
    },
    "35": {
        "inputs": {
            "garment_description": "shirt",
            "negative_prompt": "monochrome, lowres, bad anatomy, worst quality, low quality",
            "width": ["46", 1],
            "height": ["46", 2],
            "num_inference_steps": 20,
            "guidance_scale": 2.0,
            "strength": 1.0,
            "seed": 42,
            "pipeline": ["12", 0],
            "human_img": ["50", 0],
            "pose_img": ["33", 0],
            "mask_img": ["31", 0],
            "garment_img": ["15", 0]
        },
        "class_type": "IDM-VTON"
    },
    "46": {
        "inputs": {"image": ["50", 0]},
        "class_type": "GetImageSizeAndCount"
    },
    "47": {
        "inputs": {"images": ["33", 0]},
        "class_type": "PreviewImage"
    },
    "48": {
        "inputs": {"images": ["35", 0]},
        "class_type": "PreviewImage"
    },
    "50": {
        "inputs": {
            "upscale_method": "lanczos",
            "width": 624,
            "height": 880,
            "crop": "center",
            "image": ["14", 0]
        },
        "class_type": "ImageScale"
    }
}

NODE_DESCRIPTIONS = {
    "12": "Loading Pipeline", "14": "Loading Model Image", "15": "Loading Garment Image",
    "27": "Loading SAM Model", "28": "Loading GroundingDINO Model", "29": "Creating Garment Mask",
    "31": "Processing Mask", "33": "Analyzing DensePose", "35": "Applying IDM-VTON",
    "46": "Calculating Image Size", "50": "Resizing Image"
}

# ============================================================================
# COLOR HARMONY ALGORITHM
# ============================================================================

def get_dominant_color(image_array, n_clusters=5):
    """Extract dominant color using K-Means"""
    pixels = image_array.reshape(-1, 3)
    brightness = np.mean(pixels, axis=1)
    filtered_pixels = pixels[(brightness > 30) & (brightness < 225)]
    
    if len(filtered_pixels) == 0:
        filtered_pixels = pixels
    
    if len(filtered_pixels) > 10000:
        indices = np.random.choice(len(filtered_pixels), 10000, replace=False)
        filtered_pixels = filtered_pixels[indices]
    
    kmeans = KMeans(n_clusters=min(n_clusters, len(filtered_pixels)), random_state=42, n_init=10)
    kmeans.fit(filtered_pixels)
    
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    dominant_cluster = unique[np.argmax(counts)]
    dominant_color = kmeans.cluster_centers_[dominant_cluster]
    
    return tuple(map(int, dominant_color))

def rgb_to_hsv(rgb):
    """Convert RGB to HSV"""
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s, v

def calculate_color_harmony_score(image_path):
    """Calculate color harmony score"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        height, width, _ = img_array.shape
        
        center_start = int(width * 0.2)
        center_end = int(width * 0.8)
        
        top_half = img_array[:height//2, center_start:center_end]
        bottom_half = img_array[height//2:, center_start:center_end]
        
        top_color = get_dominant_color(top_half)
        bottom_color = get_dominant_color(bottom_half)
        
        top_hsv = rgb_to_hsv(top_color)
        bottom_hsv = rgb_to_hsv(bottom_color)
        
        top_hue, top_sat, top_val = top_hsv
        bottom_hue, bottom_sat, bottom_val = bottom_hsv
        
        hue_diff = abs(top_hue - bottom_hue)
        if hue_diff > 180:
            hue_diff = 360 - hue_diff
        
        top_is_neutral = top_sat < 0.2
        bottom_is_neutral = bottom_sat < 0.2
        
        score = 50
        comment = "Moderate Harmony"
        
        if top_is_neutral or bottom_is_neutral:
            score = np.random.randint(85, 96)
            comment = "Neutral Elegance"
        elif hue_diff > 150:
            score = np.random.randint(88, 98)
            comment = "Excellent Contrast"
        elif hue_diff < 30:
            score = np.random.randint(80, 92)
            comment = "Harmonious Tones"
        elif 60 < hue_diff < 120:
            score = np.random.randint(70, 85)
            comment = "Balanced Combination"
        else:
            score = np.random.randint(55, 75)
            comment = "Acceptable Match"
        
        if top_sat > 0.5 and bottom_sat > 0.5:
            score = min(100, score + 5)
        
        return score, comment, {
            "top_rgb": top_color,
            "bottom_rgb": bottom_color,
            "hue_difference": round(hue_diff, 1)
        }
    except Exception as e:
        return 0, f"Error: {str(e)}", {}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_uploaded_file(uploaded_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}_{uploaded_file.name}"
    filepath = os.path.join(COMFYUI_INPUT_PATH, filename)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filename

def update_workflow(human_filename, garment_filename, category, description):
    """Update workflow with user inputs"""
    workflow = json.loads(json.dumps(WORKFLOW_TEMPLATE))
    
    # Update file names
    workflow["14"]["inputs"]["image"] = human_filename
    workflow["15"]["inputs"]["image"] = garment_filename
    workflow["35"]["inputs"]["garment_description"] = description
    
    # HARDCODED FOR DEBUG: Force "shirt" and threshold 0.3
    workflow["29"]["inputs"]["prompt"] = "shirt"
    workflow["29"]["inputs"]["threshold"] = 0.3
    
    return workflow

def queue_prompt(workflow, client_id):
    payload = {"prompt": workflow, "client_id": client_id}
    response = requests.post(f"{COMFYUI_SERVER}/prompt", json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Queue error: {response.text}")

def get_output_images(prompt_id):
    """Try to get output image from multiple nodes + mask from node 32"""
    try:
        response = requests.get(f"{COMFYUI_SERVER}/history/{prompt_id}")
        if response.status_code != 200:
            return None, None, f"History API returned {response.status_code}"
        
        history = response.json()
        if prompt_id not in history:
            return None, None, "Prompt ID not found in history"
        
        outputs = history[prompt_id].get("outputs", {})
        
        # Get mask image from node 32 (debug)
        mask_data = None
        if "32" in outputs:
            mask_images = outputs["32"].get("images", [])
            if mask_images:
                mask_data = mask_images[0]
        
        # Try multiple output nodes for final image (35=IDM-VTON, 48=Preview, 20=Preview)
        for node_id in ["35", "48", "20"]:
            if node_id in outputs:
                images_data = outputs[node_id].get("images", [])
                if images_data:
                    return images_data[0], mask_data, None
        
        return None, mask_data, f"No images in output nodes. Available: {list(outputs.keys())}"
    except Exception as e:
        return None, None, f"Exception: {str(e)}"

def download_image_from_url(filename, subfolder="", folder_type="output"):
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(f"{COMFYUI_SERVER}/view", params=params)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    return None

def load_image_from_output_folder(filename, subfolder=""):
    """Try to find image in output or temp folders"""
    # Try output folder first
    paths_to_try = []
    
    if subfolder:
        paths_to_try.append(os.path.join(COMFYUI_OUTPUT_PATH, subfolder, filename))
        paths_to_try.append(os.path.join(COMFYUI_TEMP_PATH, subfolder, filename))
    else:
        paths_to_try.append(os.path.join(COMFYUI_OUTPUT_PATH, filename))
        paths_to_try.append(os.path.join(COMFYUI_TEMP_PATH, filename))
    
    for filepath in paths_to_try:
        if os.path.exists(filepath):
            print(f"Found image at: {filepath}")
            return Image.open(filepath)
    
    print(f"Image not found. Tried: {paths_to_try}")
    return None

# ============================================================================
# WEBSOCKET PROGRESS TRACKER
# ============================================================================

class ProgressTracker:
    def __init__(self):
        self.progress = 0
        self.max_progress = 100
        self.status = "Initializing..."
        self.completed = False
        self.error = None
        
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            if msg_type == "progress":
                self.progress = data["data"]["value"]
                self.max_progress = data["data"]["max"]
            elif msg_type == "executing":
                node_id = data["data"].get("node")
                if node_id:
                    self.status = NODE_DESCRIPTIONS.get(node_id, f"Node {node_id}")
                else:
                    self.completed = True
                    self.status = "Completed"
            elif msg_type == "execution_error":
                self.error = data["data"].get("exception_message", "Unknown error")
                self.completed = True
        except Exception as e:
            self.error = str(e)
            
    def on_error(self, ws, error):
        self.error = str(error)
        
    def on_close(self, ws, close_status_code, close_msg):
        pass
            
    def on_open(self, ws):
        self.status = "Connected"

def run_workflow_with_progress(workflow, progress_placeholder, status_placeholder):
    client_id = str(uuid.uuid4())
    tracker = ProgressTracker()
    ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
    
    ws = websocket.WebSocketApp(ws_url, on_message=tracker.on_message, on_error=tracker.on_error, on_close=tracker.on_close, on_open=tracker.on_open)
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()
    time.sleep(1)
    
    try:
        result = queue_prompt(workflow, client_id)
        prompt_id = result["prompt_id"]
        status_placeholder.info(f"Job ID: {prompt_id}")
    except Exception as e:
        ws.close()
        return None, None, None, str(e)
    
    while not tracker.completed:
        if tracker.error:
            ws.close()
            return None, None, None, tracker.error
        
        if tracker.max_progress > 0:
            progress_value = tracker.progress / tracker.max_progress
            progress_placeholder.progress(progress_value, text=tracker.status)
        else:
            progress_placeholder.progress(0, text=tracker.status)
        
        status_placeholder.text(f"Status: {tracker.status}")
        time.sleep(0.5)
    
    ws.close()
    progress_placeholder.progress(1.0, text="Completed")
    status_placeholder.success("Fetching result...")
    
    # Wait longer for file to be saved
    time.sleep(5)
    output_data, mask_data, error_msg = get_output_images(prompt_id)
    
    # Get mask image if available
    mask_image = None
    if mask_data:
        mask_filename = mask_data["filename"]
        mask_subfolder = mask_data.get("subfolder", "")
        mask_image = download_image_from_url(mask_filename, mask_subfolder)
        if mask_image is None:
            mask_image = load_image_from_output_folder(mask_filename, mask_subfolder)
    
    if output_data:
        filename = output_data["filename"]
        subfolder = output_data.get("subfolder", "")
        
        # Try API first
        image = download_image_from_url(filename, subfolder)
        image_path = None
        
        if image is None:
            # Try direct file access
            if subfolder:
                image_path = os.path.join(COMFYUI_OUTPUT_PATH, subfolder, filename)
            else:
                image_path = os.path.join(COMFYUI_OUTPUT_PATH, filename)
            
            if os.path.exists(image_path):
                image = Image.open(image_path)
            else:
                # Last resort: check for newest file in output AND temp folders
                try:
                    output_files = []
                    
                    # Search both output and temp folders
                    for search_path in [COMFYUI_OUTPUT_PATH, COMFYUI_TEMP_PATH]:
                        if os.path.exists(search_path):
                            for root, dirs, files in os.walk(search_path):
                                for file in files:
                                    if file.endswith(('.png', '.jpg', '.jpeg')):
                                        full_path = os.path.join(root, file)
                                        mtime = os.path.getmtime(full_path)
                                        output_files.append((full_path, mtime))
                    
                    if output_files:
                        # Get most recent file (created in last 5 minutes)
                        current_time = time.time()
                        recent_files = [(f, t) for f, t in output_files if (current_time - t) < 300]
                        
                        if recent_files:
                            newest_file = max(recent_files, key=lambda x: x[1])[0]
                            status_placeholder.warning(f"Using newest file: {os.path.basename(newest_file)}")
                            image = Image.open(newest_file)
                            image_path = newest_file
                        else:
                            return None, None, mask_image, f"No recent files found (last 5 min). Target: {filename}"
                    else:
                        return None, None, mask_image, f"No image files in output/temp folders"
                except Exception as e:
                    return None, None, mask_image, f"Could not find output: {str(e)}"
        else:
            temp_path = f"/tmp/{filename}"
            image.save(temp_path)
            image_path = temp_path
        
        return image, image_path, mask_image, None
    else:
        # Detailed error message
        return None, None, mask_image, f"Output not found. Details: {error_msg}"

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Virtual Try-On Studio",
        page_icon="👔",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.title("Virtual Try-On Studio")
    st.markdown('<p class="subtitle">AI-Powered Fashion Visualization</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Quick Guide")
        st.markdown("""
        **Steps:**
        1. Upload model photo
        2. Upload garment photo
        3. Select category
        4. Enter description
        5. Click Try On
        
        **Processing Time:**  
        Approximately 15-20 minutes
        """)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Test Section
        st.markdown("## Test Color Analysis")
        st.markdown("Quickly test the harmony algorithm")
        
        test_image = st.file_uploader(
            "Upload test image",
            type=["png", "jpg", "jpeg"],
            key="test_image"
        )
        
        if test_image:
            st.image(test_image, use_container_width=True)
            
            if st.button("Analyze Now"):
                with st.spinner("Processing..."):
                    temp_path = f"/tmp/test_{uuid.uuid4()}.png"
                    img = Image.open(test_image)
                    img.save(temp_path)
                    score, comment, colors = calculate_color_harmony_score(temp_path)
                    
                    st.markdown(f"""
                        <div class="test-container">
                            <div style="font-size: 3rem; font-weight: 700; text-align: center; color: #000000;">{score}</div>
                            <div style="font-size: 1.2rem; font-weight: 600; text-align: center; color: #000000; margin-top: 0.5rem;">{comment}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if colors:
                        st.caption(f"Top: RGB{colors['top_rgb']}")
                        st.caption(f"Bottom: RGB{colors['bottom_rgb']}")
                        st.caption(f"Hue Difference: {colors['hue_difference']}°")
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.caption(f"Server: {COMFYUI_SERVER}")
    
    # Main Section
    st.markdown("## Upload Images")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("**Model Photo**")
        human_file = st.file_uploader("Model", type=["png", "jpg", "jpeg"], key="human", label_visibility="collapsed")
        if human_file:
            st.image(human_file, use_container_width=True)
    
    with col2:
        st.markdown("**Garment Photo**")
        garment_file = st.file_uploader("Garment", type=["png", "jpg", "jpeg"], key="garment", label_visibility="collapsed")
        if garment_file:
            st.image(garment_file, use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## Configuration")
    
    col3, col4 = st.columns([1, 2], gap="large")
    
    with col3:
        category = st.radio("Garment Category", ["Upper Body", "Lower Body"])
    
    with col4:
        description = st.text_input(
            "Garment Description",
            placeholder="e.g., Red polo shirt, Blue denim jeans",
            value="Red polo shirt"
        )
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Try On Button
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        try_on_button = st.button("TRY ON", type="primary", use_container_width=True)
    
    if try_on_button:
        if not human_file or not garment_file:
            st.error("Please upload both model and garment images")
            return
        
        if not description.strip():
            st.error("Please enter garment description")
            return
        
        if not os.path.exists(COMFYUI_INPUT_PATH):
            st.error(f"Input folder not found: {COMFYUI_INPUT_PATH}")
            return
        
        with st.spinner("Uploading files..."):
            human_filename = save_uploaded_file(human_file)
            garment_filename = save_uploaded_file(garment_file)
        
        st.success("Files uploaded successfully")
        
        workflow = update_workflow(human_filename, garment_filename, category, description)
        st.info(f"Processing with HARDCODED mask: **shirt** (threshold: 0.3) - Debug Mode")
        
        st.markdown("## Processing")
        
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        result_image, image_path, mask_image, error = run_workflow_with_progress(workflow, progress_placeholder, status_placeholder)
        
        if error:
            st.error(f"Error: {error}")
        elif result_image:
            st.success("Try-On completed successfully")
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Debug: Show mask if available
            if mask_image:
                st.markdown("## Debug: Mask")
                st.image(mask_image, caption="Generated Mask (Node 32)", use_container_width=True)
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Color Analysis
            st.markdown("## Color Harmony Analysis")
            
            with st.spinner("Analyzing color harmony..."):
                score, comment, colors = calculate_color_harmony_score(image_path)
            
            col_score = st.columns([1, 2, 1])
            with col_score[1]:
                st.markdown(f"""
                    <div class="score-container">
                        <div class="score-number">{score}</div>
                        <div class="score-text">{comment}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Results
            st.markdown("## Results")
            
            col_r1, col_r2, col_r3 = st.columns(3, gap="large")
            
            with col_r1:
                st.markdown("**Original Model**")
                st.image(human_file, use_container_width=True)
            
            with col_r2:
                st.markdown("**Garment**")
                st.image(garment_file, use_container_width=True)
            
            with col_r3:
                st.markdown("**Result**")
                st.image(result_image, use_container_width=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Download
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            
            col_dl = st.columns([1, 2, 1])
            with col_dl[1]:
                st.download_button(
                    label="DOWNLOAD RESULT",
                    data=buf.getvalue(),
                    file_name=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
        else:
            st.warning("Could not retrieve result")

if __name__ == "__main__":
    main()
