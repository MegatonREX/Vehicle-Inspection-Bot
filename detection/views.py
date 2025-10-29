import os
import tempfile
import numpy as np
import cv2
import requests
import json
import re
import logging
import base64
import time
from django.shortcuts import render
from django.http import FileResponse, HttpResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
import shutil
from typing import Dict, Any, Optional
from ultralytics import YOLO
import random

# Create base directories for saving media files
MEDIA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "media")
# Create base directories for saving media files
MEDIA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "media")
PROCESSED_IMAGES_DIR = os.path.join(MEDIA_ROOT, "360_processed_images")
PROCESSED_VIDEOS_DIR = os.path.join(MEDIA_ROOT, "uploaded_360_videos")

# Create all required directories if they don't exist
os.makedirs(MEDIA_ROOT, exist_ok=True)
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_temp_directory():
    """Create a temporary directory for frame storage"""
    return tempfile.mkdtemp(prefix='video_frames_')

def cleanup_temp_files(temp_dir, temp_files=None):
    """Clean up temporary files and directory"""
    if temp_files:
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.debug(f"Removed temp file: {file_path}")
            except Exception as e:
                logging.error(f"Error removing temp file {file_path}: {e}")
    
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logging.debug(f"Removed temp directory: {temp_dir}")
        except Exception as e:
            logging.error(f"Error removing temp directory: {e}")

def save_frame_temp(frame, temp_dir, frame_idx):
    """Save a frame to temporary storage"""
    frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.jpg")
    cv2.imwrite(frame_path, frame)
    return frame_path

def serve_video(request, filename):
    """
    Serve processed videos from the processed_videos directory
    """
    try:
        # Security check: ensure filename doesn't contain path traversal
        if '..' in filename or '/' in filename:
            return HttpResponse("Invalid filename", status=400)

        video_path = os.path.join(PROCESSED_VIDEOS_DIR, filename)
        if not os.path.exists(video_path):
            return HttpResponse("Video not found", status=404)
        
        # Verify file extension
        ext = os.path.splitext(filename)[1].lower()
        content_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
        }
        
        if ext not in content_types:
            return HttpResponse("Unsupported video format", status=400)

        # Open the file in binary mode and create response
        video_file = open(video_path, 'rb')
        response = FileResponse(video_file)
        response['Content-Type'] = content_types[ext]
        response['Content-Disposition'] = f'inline; filename="{filename}"'
        return response

    except IOError:
        return HttpResponse("Error reading video file", status=500)
    except Exception as e:
        return HttpResponse(f"Error serving video: {str(e)}", status=500)

def serve_video(request, filename):
    """
    Serve processed videos from the processed_videos directory
    """
    try:
        # Security check: ensure filename doesn't contain path traversal
        if '..' in filename or '/' in filename:
            return HttpResponse("Invalid filename", status=400)

        video_path = os.path.join(PROCESSED_VIDEOS_DIR, filename)
        if not os.path.exists(video_path):
            return HttpResponse("Video not found", status=404)
        
        # Verify file extension
        ext = os.path.splitext(filename)[1].lower()
        content_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
        }
        
        if ext not in content_types:
            return HttpResponse("Unsupported video format", status=400)

        # Open the file in binary mode and create response
        video_file = open(video_path, 'rb')
        response = FileResponse(video_file)
        response['Content-Type'] = content_types[ext]
        response['Content-Disposition'] = f'inline; filename="{filename}"'
        return response

    except IOError:
        return HttpResponse("Error reading video file", status=500)
    except Exception as e:
        return HttpResponse(f"Error serving video: {str(e)}", status=500)

# ======================
# MODEL LOADING
# ======================
CLS_MODEL = YOLO(r"D:/work/motorDNA/vehicle-inspection/model/best.pt")

# Auto-sync class names from YOLO model
CLASS_NAMES = list(CLS_MODEL.names.values())

CLASS_NAME_MAP = {
    "gear stick": "gear_stick",
    "steering wheel": "steering_wheel",
    "rear": "rear",
    "front": "front",
    "left": "left",
    "right": "right",
    "seats": "seats",
    "dashboard": "dashboard",
    "console": "console"
}


# =======================
# API Configuration
# =======================
OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT = int(os.environ.get('TIMEOUT', '30'))

# =======================
# PROMPTS AND JSON KEYS
# =======================
class_prompts = {
    "dashboard": {
        "System": (
            "You are a vehicle dashboard analyst. Look at the dashboard image and extract all details, "
            "Including temperature, Trip_meter, Gear_position, Fuel, Dashboard_condition, and Extra details. "
            "For warning_lights: "
            "- Only list icons that are visibly turned on in the image. "
            "- Do NOT infer or guess warning lights that are not visible. "
            "- Use descriptive names if exact match is unclear (e.g., 'seatbelt icon', 'door open icon', 'engine icon'). "
            "- Exclude regular gauges (like temperature or fuel) unless they show a warning state. "
            "- If no warning lights are on, return 'none'."
        ),
        "User": "Please analyze this dashboard image.",
        "Keys": ["Temperature", "Trip_meter", "Warning_lights", "Fuel", "Dashboard_condition"]
        # "System": (
        #     "You are an expert vehicle dashboard analyst. Examine the dashboard image carefully "
        #     "and provide a structured summary of all observable details. Focus on extracting the following information accurately "
        #     "door_status(whether the doors are open or closed)"
        #     "temperature, trip meter reading, gear position, list of warning lights in the dashboard image,fuel level, overall dashboard condition, "
        #     "and any additional relevant details. "
        #     "Provide the output in a clear, structured format corresponding to these keys: "
        #     "temperature, trip_meter, gear_position, warning_lights, door_status fuel, dashboard_condition."
        # ),
        # "User": "Please analyze this dashboard image and provide all details.",
        # "Keys": ["temperature", "trip_meter", "gear_position", "warning_lights", "door_status", "fuel", "dashboard_condition"]
    },
    "console": {
        "System": (
            "You are a vehicle console analyst. Look at the console image and return JSON with ONLY these keys: "
            "Infotainment_system, Climate_control, Air_vents, Shift_lever, Cup_holders, Other_features, Condition. "
            "Use 'unknown' for missing values. Respond ONLY with JSON. "
            "Provide answers as keywords or short phrases, not full sentences. Examples: "
            "'touchscreen', 'automatic','manual', 'adjustable','1','2', '2+', 'storage', steering_controls', 'excellent','good','damaged', 'dirty','Present'."
        ),
        "User": "Please analyze this console image.",
        "Keys": ["Infotainment_system", "Climate_control", "Air_vents","Condition", "Other_features"]
    },
    "steering_wheel": {
        "System": (
            "You are a vehicle steering wheel analyst. Look at the steering wheel image and return JSON with ONLY these keys: "
            "Steering_type, Steering_condition. "
            "Use 'unknown' if a value is missing. Respond ONLY with JSON. "
            "Provide answers as keywords or short phrases, not full sentences. Examples: "
            "'manual', 'power', 'ok', 'damaged', 'worn', 'dirty'."
        ),
        "User": "Please analyze this steering wheel image.",
        "Keys": ["Steering_type", "Steering_condition"]
    },
    "seats": {
        "System": (
            "You are a vehicle seats analyst. Look at the seats image and return JSON with ONLY these keys: "
            "Seat_material, Seat_color, Seat_condition. "
            "Use 'unknown' if a value is missing. Respond ONLY with JSON. "
            "Provide answers as keywords or short phrases, not full sentences. Examples: "
            "'leather', 'fabric', 'black', 'grey', 'blue', 'excellent', 'good', 'damaged', 'dirty'."
        ),
        "User": "Please analyze this seats image.",
        "Keys": ["Seat_material", "Seat_color", "Seat_condition"]
    },
    "gear_stick": {
        "System": (
            "You are a vehicle gear stick analyst. Look at the gear stick image and return JSON with ONLY these keys: "
            "Gear_position, Gear_type(manual/automatic), Gear_stick_condition. "
            "Use 'unknown' if a value is missing. Respond ONLY with JSON. "
            "Provide answers as keywords or short phrases, not full sentences. Examples: "
            "'P', 'R', 'N', 'D', '1', '2', 'manual', 'automatic', 'ok', 'damaged', 'dirty'."
        ),
        "User": "Please analyze this gear stick image.",
        "Keys": ["Gear_type", "Gear_stick_condition"]
    },
    "front": {
        "System": (
            "You are a vehicle front view analyst. Look at the front image and return JSON with ONLY these keys: "
            "Brand, Model, Year, Colour, License_plate (detected/not detected/damaged), "
            "Headlight_status (ok/damaged/missing), Grille_condition (ok/damaged/missing), "
            "Bumper_condition (ok/damaged/missing), Windshield_condition (ok/damaged/cracked/missing), "
            "Paint_quality (ok/bad/scratched/rusted), Emblem/logo_condition (ok/damaged/not detected/missing). "
            "If a detail is clearly visible, extract it. If it cannot be determined, use 'unknown'. "
            "Respond ONLY with JSON, using short keywords or phrases (e.g., 'ok', 'damaged', 'dirty', 'missing')."
        ),
        "User": "Please analyze this front image.",
        "Keys": ["Brand", "Model", "Year", "Colour", "Paint_quality", "License_plate","Emblem/logo_condition","Headlight_status", "Grille_condition", "Bumper_condition", "Windshield_condition"]
    },
    "left": {
        "System": (
            "You are a vehicle left side view analyst. Look at the left side image and return JSON with ONLY these keys: "
            "Brand, Model, Year, Door_condition (ok/damaged/dented/scratched/missing), "
            "Mirror_status (ok/damaged/missing), Window_status (ok/damaged/cracked/missing/not detected), "
            "Left_side_condition (ok/dent/scratch/damaged). "
            "If a detail is clearly visible, extract it. If it cannot be determined, use 'unknown'. "
            "Respond ONLY with JSON, using short keywords or phrases (e.g., 'ok', 'damaged', 'dirty', 'missing')."
        ),
        "User": "Please analyze this left side image.",
        "Keys": ["Brand", "Model", "Year", "Door_condition", "Mirror_status", "Window_status", "Left_side_condition"]
    },
    "right": {
        "System": (
            "You are a vehicle right side view analyst. Look at the right side image and return JSON with ONLY these keys: "
            "Brand, Model, Year, Door_condition (ok/damaged/dented/scratched/missing), "
            "Mirror_status (ok/damaged/missing), Window_status (ok/damaged/cracked/missing), "
            "Right_side_condition (ok/dent/scratch/damaged). "
            "If a detail is clearly visible, extract it. If it cannot be determined, use 'unknown'. "
            "Respond ONLY with JSON, using short keywords or phrases (e.g., 'ok', 'damaged', 'dirty', 'missing')."
        ),
        "User": "Please analyze this right side image.",
        "Keys": ["Brand", "Model", "Year", "Door_condition", "Mirror_status", "Window_status", "Right_side_condition"]
    },
    "rear": {
        "System": (
            "You are a vehicle rear view analyst. Look at the rear image and return JSON with ONLY these keys: "
            "Brand, Model, Year, Colour, License_plate (detected/not detected/damaged), "
            "Taillight_status (damaged/not damaged/missing), Bumper_condition (damaged/not damaged/missing), "
            "Trunk_status (damaged/not damaged/open/closed/missing), Rear_condition (damaged/not damaged), "
            "Rear_windshield_condition (not detected/damaged/not damaged), Paint_quality (ok/bad/scratched/rusted), "
            "Emblem/Logo_condition (ok/damaged/not detected/missing). "
            "If a detail is clearly visible, extract it. If it cannot be determined, use 'unknown'. "
            "Respond ONLY with JSON, using short keywords or phrases (e.g., 'ok', 'damaged', 'dirty', 'missing')."
        ),
        "User": "Please analyze this rear image.",
        "Keys": ["Brand", "Model", "Year", "Colour","Paint_quality","License_plate","Emblem/Logo_condition","Taillight_status", "Bumper_condition", "Rear_condition", "Rear_windshield_condition"]
    }
}

# Define the 4 required views for 360-degree completeness
REQUIRED_360_VIEWS = {"front", "left", "right", "rear"}

# All valid views that can be detected (for validation)
VALID_VIEWS = {cls for cls in CLASS_NAMES if cls not in ["misc"]}

# =======================
# Utility Functions
# =======================
def encode_image(image_path: str) -> str:
    """Encode image to base64 for API request."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_prompt_and_keys(frame_class: str):
    """Get the appropriate prompts and expected JSON keys for a class."""
    cls = class_prompts.get(frame_class, class_prompts["dashboard"])
    return cls["System"], cls["User"], cls["Keys"]

def clean_json_string(text: str) -> str:
    """Clean JSON string by removing code fences and fixing formatting."""
    text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    return text

def analyze_frame(frame_path: str, frame_class: str) -> Dict[str, Any]:
    """Analyze a single frame using the Ollama API."""
    try:
        system_prompt, user_question, json_keys = get_prompt_and_keys(frame_class)
        image_b64 = encode_image(frame_path)
        
        if frame_class == "dashboard":
            # Special handling for dashboard with multiple analyses
            all_responses = []

            # Run analysis n times and collect responses
            for _ in range(1):
                initial_payload = {
                "model": "gemma3:4b",
                "prompt": f"System: {system_prompt}\nUser: {user_question}",
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0,
                            }
                }   
                response = requests.post(OLLAMA_URL, json=initial_payload).json()
                text_response = response.get("response", "").strip()
                cleaned_text = clean_json_string(text_response)
                if cleaned_text:  # Only add non-empty responses
                    all_responses.append(cleaned_text)

            # Combine all responses for the JSON parser
            combined_responses = " ".join(all_responses)
            # print(combined_responses)
            
            # Second analysis to clean up the JSON and extract unique answers
            payload = {
                    "model": "llama3:8b",
                    "prompt": (
                        f"System: You are a JSON parser. Analyze these multiple responses and extract ONLY these keys: {json_keys}. "
                        f"If multiple values exist for a key, use the most specific. "
                        f"Use 'unknown' for missing keys. Respond ONLY with JSON.\n"
                        f"User: Here are the responses: {combined_responses}"
                    ),
                    "stream": False,
                    "options": {"temperature": 0}
                }

            response = requests.post(OLLAMA_URL, json=payload).json()
            text_response = response.get("response", "").strip()
            cleaned_text = clean_json_string(text_response)

        else:
            # Standard analysis for other views
            payload = {
                "model": "gemma3:4b",
                "prompt": f"System: {system_prompt}\nUser: {user_question}",
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0}
            }
            
            response = requests.post(OLLAMA_URL, json=payload).json()
            text_response = response.get("response", "").strip()
            cleaned_text = clean_json_string(text_response)
        
        # Parse JSON for all cases
        try:
            raw_data = json.loads(cleaned_text)
        except json.JSONDecodeError:
            logging.warning("Failed to parse JSON response. Using default values.")
            raw_data = {}
        
        # Ensure only requested keys are included
        result_data = {k: raw_data.get(k, "unknown") for k in json_keys}
        return result_data
        
    except Exception as e:
        logging.error(f"Error analyzing frame: {str(e)}")
        return {k: "unknown" for k in get_prompt_and_keys(frame_class)[2]}

def get_default_structure(frame_class: str) -> Dict[str, str]:
    """Get default structure based on frame class."""
    defaults = {
        "dashboard": {
            "speed": "unknown",
            "rpm": "unknown", 
            "temperature": "unknown",
            "odometer": "unknown",
            "trip_meter": "unknown",
            "warning_lights": "unknown"
        },
        "gear_stick": {
            "gear_position": "unknown",
            "gear_type": "unknown",
            "gear_stick_condition": "unknown"
        },
        "seats": {
            "seat_material": "unknown",
            "seat_color": "unknown", 
            "seat_condition": "unknown"
        },
        "steering_wheel": {
            "steering_type": "unknown",
            "steering_condition": "unknown"
        },
        "console": {
            "console_features": ["unknown"],
            "console_condition": "unknown"
        }
    }
    
    exterior_defaults = {
        "brand": "unknown",
        "model": "unknown",
        "year": "unknown",
    }

    if frame_class == "front":
        exterior_defaults.update({
            "front_damage": "unknown",
            "front_damage_part": "unknown",
            "front_damage_type": "unknown"
        })
    elif frame_class == "rear":
        exterior_defaults.update({
            "rear_damage": "unknown",
            "rear_damage_part": "unknown",
            "rear_damage_type": "unknown"
        })
    elif frame_class == "left":
        exterior_defaults.update({
            "left_damage": "unknown",
            "left_damage_part": "unknown",
            "left_damage_type": "unknown"
        })
    elif frame_class == "right":
        exterior_defaults.update({
            "right_damage": "unknown",
            "right_damage_part": "unknown",
            "right_damage_type": "unknown"
        })
    elif frame_class == "general":
        exterior_defaults.update({
            "damage": "unknown"
        })
    
    if frame_class in ["front", "rear", "left", "right", "general"]:
        return exterior_defaults
    
    return defaults.get(frame_class, exterior_defaults)

# File extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


# ======================
# API Processing Function
# ======================
def process_frame_api(frame_file: str, frame_path: str, frame_class: str = "general") -> tuple[Optional[Dict[str, Any]], str]:
    """Process a frame with class-specific prompts based on model prediction"""
    save_path = ""
    try:
        # First normalize the class name to match our prompt keys
        frame_class = frame_class.lower()
        frame_class = CLASS_NAME_MAP.get(frame_class, frame_class)

        # Create a directory to save processed images if it doesn't exist
        save_dir = os.path.join(PROCESSED_IMAGES_DIR, frame_class)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the image with a unique name including timestamp and ensure unique filename
        timestamp = int(time.time())
        base_name = os.path.splitext(os.path.basename(frame_file))[0]
        # Clean the base_name to remove any potentially problematic characters
        base_name = re.sub(r'[^\w\-_]', '', base_name)
        save_path = os.path.join(save_dir, f"{frame_class}_{timestamp}_{base_name}.jpg")
        
        # Ensure the file doesn't already exist
        counter = 1
        while os.path.exists(save_path):
            save_path = os.path.join(save_dir, f"{frame_class}_{timestamp}_{base_name}_{counter}.jpg")
            counter += 1
            
        # Copy the image to the save location
        shutil.copy2(frame_path, save_path)
        logging.info(f"➡️ Processing frame: {frame_path} (class={frame_class})")
        logging.info(f"💾 Saved processed image to: {save_path}")

        # Get YOLO model prediction to confirm class
        results = CLS_MODEL.predict(frame_path, verbose=False)
        result = results[0]
        
        if result and result.probs is not None:
            probs = result.probs.data.cpu().numpy()
            idx = int(np.argmax(probs))
            predicted_class = CLASS_NAMES[idx].lower() if idx < len(CLASS_NAMES) else "general"
            predicted_class = CLASS_NAME_MAP.get(predicted_class, predicted_class)
            confidence = float(probs[idx] * 100.0)

            if predicted_class in class_prompts:
                frame_class = predicted_class
                logging.info(f"Using normalized predicted class {predicted_class} ({confidence:.2f}%) for analysis")
            else:
                logging.warning(f"Using provided class {frame_class} instead of predicted {predicted_class}")

        else:
            logging.warning(f"No prediction result, using provided class {frame_class}")

        # Analyze frame with Ollama API
        result_json = analyze_frame(frame_path, frame_class)
        
        if result_json:
            logging.info("✅ Successfully analyzed frame")
            logging.info("🚗 Car Analysis Results (JSON format):")
            logging.info(json.dumps(result_json, indent=2))
            return result_json, save_path
        else:
            logging.warning("⚠️ Analysis failed, using default values")
            _, _, json_keys = get_prompt_and_keys(frame_class)
            return {k: "unknown" for k in json_keys}, save_path
            
    except Exception as e:
        logging.error(f"❌ API processing error for {frame_file}: {str(e)}")
        _, _, json_keys = get_prompt_and_keys(frame_class)
        return {k: "unknown" for k in json_keys}, save_path
    finally:
        logging.info("-" * 50)

# ======================
# CLASSIFICATION FUNCTION
# ======================
def classify_view(img_path):
    try:
        results = CLS_MODEL.predict(img_path, verbose=False)
        result = results[0]

        # No classification output
        if not result or result.probs is None:
            return None, None

        # Probabilities
        probs = result.probs.data.cpu().numpy()
        idx = int(np.argmax(probs))

        # Safety check
        if idx >= len(CLASS_NAMES):
            return None, None

        label = CLASS_NAMES[idx]
        conf = float(probs[idx] * 100.0)
        return label, conf

    except Exception as e:
        print("Error in classify_view:", e)
        return None, None


# ======================
# VIDEO PROCESSING FUNCTION
# ======================
def extract_frames_from_video(video_path, skip_frames=15, batch_size=10):
    """
    Extract frames from video in batches to reduce memory usage.
    Returns tuple of (frame_paths, temp_dir).
    """
    temp_dir = create_temp_directory()
    frame_paths = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}")
        return [], temp_dir
    
    frame_count = 0
    extracted_count = 0
    batch_frames = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames according to skip_frames parameter
            if frame_count % (skip_frames + 1) == 0:
                # Save frame to temporary directory
                frame_path = save_frame_temp(frame, temp_dir, extracted_count)
                frame_paths.append(frame_path)
                batch_frames.append((frame, frame_path))
                extracted_count += 1
                
                # Process batch when it reaches batch_size
                if len(batch_frames) >= batch_size:
                    # Clear batch after processing
                    batch_frames = []
                    
                # Log progress periodically
                if extracted_count % 50 == 0:
                    logging.info(f"Extracted {extracted_count} frames...")
            
            # Clear frame from memory
            frame = None
            frame_count += 1
            
    except Exception as e:
        logging.error(f"Error during frame extraction: {e}")
        cleanup_temp_files(temp_dir, frame_paths)
        return [], temp_dir
        
    finally:
        cap.release()
        
    logging.info(f"Extracted {len(frame_paths)} frames from video (skipped every {skip_frames} frames)")
    return frame_paths, temp_dir


# ======================
# PROCESSING FUNCTION FOR FRAMES
# ======================
def process_frames(frame_paths, source_name="video"):
    """
    Process extracted frames and return classification results.
    Groups by class and keeps the highest confidence for each class.
    """
    class_results = {}  # {class_name: {"confidence": max_conf, "frame_info": frame_details}}
    all_detections = []
    
    for i, frame_path in enumerate(frame_paths):
        label, conf = classify_view(frame_path)
        
        if label is None or label not in VALID_VIEWS:
            all_detections.append({
                "frame": f"frame_{i}",
                "source": source_name,
                "result": "Could not classify"
            })
            continue
        
        detection_info = {
            "frame": f"frame_{i}",
            "source": source_name,
            "result": f"{label} ({conf:.2f}%)"
        }
        all_detections.append(detection_info)
        
        # Keep the highest confidence detection for each class
        if label not in class_results or conf > class_results[label]["confidence"]:
            class_results[label] = {
                "confidence": conf,
                "frame_info": detection_info
            }
    
    return class_results, all_detections

def calculate_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    return cv2.Laplacian(image, cv2.CV_64F).var()

# ======================
# Frontend View
# ======================
def index(request):
    """Render the index page."""
    return render(request, 'index.html')

# ======================
# API ENDPOINT
# ======================
@api_view(['POST'])
def check_360(request):
    # Get all uploaded files
    all_files = []
    for field_name in request.FILES:
        files = request.FILES.getlist(field_name)
        all_files.extend(files)
    
    if not all_files:
        return Response({
            "error": "Please upload either image files or a video file"
        }, status=400)

    # Separate videos and images
    video_files = []
    image_files = []
    
    for file_obj in all_files:
        file_name = getattr(file_obj, 'name', 'unknown_file')
        file_ext = os.path.splitext(file_name)[1].lower()
        
        if file_ext in VIDEO_EXTENSIONS:
            video_files.append(file_obj)
        elif file_ext in IMAGE_EXTENSIONS:
            image_files.append(file_obj)

    # Check: either video or images, not both
    if video_files and image_files:
        return Response({
            "error": "Please upload either a video OR images, not both"
        }, status=400)

    if not video_files and not image_files:
        return Response({
            "error": "Please upload supported file formats. Videos: " + ', '.join(VIDEO_EXTENSIONS) + 
                     " Images: " + ', '.join(IMAGE_EXTENSIONS)
        }, status=400)

    detected_views = set()
    summary = []
    temp_paths = []
    processing_mode = None
    
    try:
        # Process video upload
        if video_files:
            if len(video_files) > 1:
                return Response({
                    "error": "Please upload only one video file at a time"
                }, status=400)
            
            processing_mode = "video"
            video_file = video_files[0]
            
            # Save video to a permanent location
            timestamp = int(time.time())
            # Clean the filename to remove any potentially problematic characters
            original_name = os.path.splitext(video_file.name)[0]
            clean_name = re.sub(r'[^\w\-_]', '', original_name)
            file_ext = os.path.splitext(video_file.name)[1].lower()
            video_filename = f"video_{timestamp}_{clean_name}{file_ext}"
            permanent_video_path = os.path.join(PROCESSED_VIDEOS_DIR, video_filename)
            
            # Save the video file
            with open(permanent_video_path, 'wb') as video_dest:
                for chunk in video_file.chunks():
                    video_dest.write(chunk)
            
            # Create temporary copy for processing
            file_ext = os.path.splitext(video_file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_video:
                shutil.copy2(permanent_video_path, tmp_video.name)
                temp_video_path = tmp_video.name
                temp_paths.append(temp_video_path)
            
            # Extract frames from video
            frame_paths, temp_dir = extract_frames_from_video(temp_video_path, skip_frames=3, batch_size=10)
            if temp_dir:
                temp_paths.append(temp_dir)
            temp_paths.extend(frame_paths)
            
            if not frame_paths:
                return Response({
                    "error": "Could not extract frames from video"
                }, status=400)
            
            best_frames = {}  # {class: (frame_path, conf, combined_score, idx)}
            batch_size = 10  # Process frames in smaller batches
            
            # Process frames in batches to reduce memory usage
            for i in range(0, len(frame_paths), batch_size):
                batch = frame_paths[i:i + batch_size]
                
                for frame_idx, frame_path in enumerate(batch, i):
                    if not os.path.exists(frame_path):
                        continue
                        
                    label, conf = classify_view(frame_path)
                    if label and label in VALID_VIEWS and conf > 85:
                        sharpness = calculate_sharpness(frame_path)
                        combined_score = conf * sharpness if conf is not None else 0
                        
                        if label not in best_frames or combined_score > best_frames[label][2]:
                            # Remove previous best frame for this label if it exists
                            if label in best_frames:
                                old_frame_path = best_frames[label][0]
                                if old_frame_path in temp_paths:
                                    try:
                                        os.remove(old_frame_path)
                                        temp_paths.remove(old_frame_path)
                                    except:
                                        pass
                            
                            best_frames[label] = (frame_path, conf, combined_score, frame_idx)
                    else:
                        # Remove unneeded frame
                        try:
                            os.remove(frame_path)
                            if frame_path in temp_paths:
                                temp_paths.remove(frame_path)
                        except:
                            pass
                        
            # Process only the best frame per class through the API
            summary = []
            for label, (frame_path, conf, combined_score, i) in best_frames.items():
                frame_class = label
                api_result, saved_image_path = process_frame_api(f"frame_{i}.jpg", frame_path, frame_class)
                summary.append({
                    "frame": f"frame_{i}",
                    "class": frame_class,
                    "classification": label,
                    "confidence": conf,
                    "api_result": api_result,
                    "saved_image_path": saved_image_path
                })
                detected_views.add(label)
        
        # Process image uploads
        else:
            processing_mode = "images"
            
            if len(image_files) < 1:
                return Response({
                    "error": f"At least 4 images are required. You uploaded {len(image_files)} image(s)"
                }, status=400)
            
            # Classify all images, keep only highest confidence per class
            best_images = {}  # {class: (temp_path, confidence, file_name)}
            for idx, file_obj in enumerate(image_files):
                file_name = getattr(file_obj, 'name', f'image_{idx}.jpg')
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                        for chunk in file_obj.chunks():
                            temp_file.write(chunk)
                        temp_path = temp_file.name
                    temp_paths.append(temp_path)
                    # label, conf = classify_view(temp_path)
                    # if label and label in VALID_VIEWS and conf > 80:
                    #     if label not in best_images or conf > best_images[label][1]:
                    #         best_images[label] = (temp_path, conf, file_name)
                    label, conf = classify_view(temp_path)
                    sharpness = calculate_sharpness(temp_path)
                    combined_score = conf * sharpness if conf is not None else 0

                    if label and label in VALID_VIEWS and conf > 85:
                        if label not in best_images or combined_score > best_images[label][2]:
                            best_images[label] = (temp_path, conf, combined_score, file_name)
                except Exception as e:
                    summary.append({
                        "image": file_name,
                        "error": str(e)
                    })
            
            # Process only the best image per class through the API
            for label, (temp_path, conf, combined_score, file_name) in best_images.items():
                frame_class = label
                api_result, saved_image_path = process_frame_api(file_name, temp_path, frame_class)
                summary.append({
                    "image": file_name,
                    "class": frame_class,
                    "classification": label,
                    "confidence": conf,
                    "api_result": api_result,
                    "saved_image_path": saved_image_path
                })
                detected_views.add(label)

        # Check specifically for 360-degree completeness (front, left, right, rear)
        detected_360_views = detected_views.intersection(REQUIRED_360_VIEWS)
        missing_360_views = list(REQUIRED_360_VIEWS - detected_360_views)
        is_complete_360 = len(detected_360_views) == 4  # All 4 required views present

        response_data = {
            "processing_mode": processing_mode,
            "detected_views": list(detected_views),
            "detected_360_views": list(detected_360_views),
            "missing_360_views": missing_360_views,
            "is_complete_360": is_complete_360,
            "details": summary
        }

        # Add video URL if video was processed
        if video_files:
            video_filename = os.path.basename(permanent_video_path)
            response_data["video_url"] = f"/media/videos/{video_filename}"

        return Response(response_data)

    finally:
        # Cleanup all temporary files
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {temp_path}: {e}")