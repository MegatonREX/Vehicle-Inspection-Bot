# 🚗 Vehicle Inspection Bot

A Django-based AI-powered web application that analyzes vehicle images and videos to detect complete 360° coverage and classify different car views using YOLO object detection and Ollama AI for detailed inspection reports.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [How It Works](#how-it-works)
- [Valid View Types](#valid-view-types)
- [Technologies Used](#technologies-used)
- [Configuration Parameters](#configuration-parameters)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Multi-Format Support**: Upload multiple images or a single video for comprehensive vehicle analysis
- **360° Coverage Detection**: Automatically verifies if uploaded media provides complete 360° vehicle coverage
- **Intelligent View Classification**: Identifies and classifies 9 different car views using YOLO:
  - **Exterior**: Front, Rear, Left, Right
  - **Interior**: Dashboard, Console, Seats, Steering Wheel, Gear Stick
- **AI-Powered Detailed Analysis**: Uses Ollama API to extract detailed information from each view including:
  - Vehicle condition assessment
  - Damage detection
  - Component status
  - Brand, model, and year identification
- **Smart Frame Selection**: Extracts optimal frames from videos based on:
  - YOLO confidence scores
  - Image sharpness metrics
  - Combined quality scoring
- **Real-time Processing**: Interactive web interface with drag-and-drop file upload
- **Comprehensive Reports**: Provides detailed analysis with confidence scores and missing views identification
- **Cross-Platform Compatible**: Works on both Windows and Linux systems

## 📁 Project Structure

```
vehicle-inspection/
├── car360/                      # Main Django project
│   ├── settings.py             # Django settings configuration
│   ├── urls.py                 # Main URL routing
│   ├── wsgi.py                 # WSGI configuration
│   ├── asgi.py                 # ASGI configuration
│   └── train/                  # YOLO training configuration
│       ├── args.yaml           # Training arguments
│       ├── results.csv         # Training results
│       └── weights/            # Model weights
│           ├── best.pt         # Best model checkpoint
│           └── last.pt         # Latest model checkpoint
├── detection/                   # Detection app
│   ├── views.py                # Main logic and API endpoints
│   ├── urls.py                 # App URL configuration
│   ├── models.py               # Database models
│   ├── admin.py                # Django admin configuration
│   └── migrations/             # Database migrations
├── templates/
│   └── index.html              # Frontend interface
├── media/                      # Media storage
│   ├── 360_processed_images/  # Processed and classified images
│   └── uploaded_360_videos/   # Uploaded video files
├── model/                      # YOLO model files
│   ├── best.pt                # Primary YOLO model
│   └── best1.pt               # Secondary model
├── requirements.txt            # Python dependencies
├── manage.py                   # Django management script
├── db.sqlite3                  # SQLite database
└── README.md                   # This file
```

## 🔧 Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended for faster processing)
- **Ollama** installed and running
- **FFmpeg** (for video processing)
- **Operating System**: Windows or Linux

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/MegatonREX/Vehicle-Inspection-Bot.git
cd vehicle-inspection
```

### 2. Create a virtual environment

**On Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Configure Ollama

**Download and install Ollama:**
- Visit [Ollama website](https://ollama.ai/)
- Download the installer for your platform
- Install Ollama

**Start Ollama service:**
```bash
ollama serve
```

**Pull a vision model (recommended):**
```bash
ollama pull llava
```

### 5. Install FFmpeg

**On Windows:**
- Download from [FFmpeg website](https://ffmpeg.org/download.html)
- Add to system PATH

**On Linux:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**On Mac:**
```bash
brew install ffmpeg
```

### 6. Set up the database

```bash
python manage.py migrate
```

### 7. Create required directories

```bash
mkdir media\360_processed_images media\uploaded_360_videos  # Windows
# OR
mkdir -p media/360_processed_images media/uploaded_360_videos  # Linux/Mac
```

## ⚙️ Configuration

### Model Configuration

Update the model path in `detection/views.py` to use a relative path:

```python
# For cross-platform compatibility
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
CLS_MODEL = YOLO(MODEL_PATH)
```

### Ollama Configuration

By default, the system uses Ollama API running on `http://localhost:11434`. 

To change the Ollama URL, update `detection/views.py`:

```python
OLLAMA_URL = "http://localhost:YOUR_PORT/api/generate"
```

Or use environment variables:

```bash
export OLLAMA_URL="http://your-ollama-host:11434/api/generate"
export TIMEOUT="60"
```

### Django Settings

In `car360/settings.py`, verify:

```python
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

ALLOWED_HOSTS = ['localhost', '127.0.0.1']  # Add your domain in production
```

## 🚀 Usage

### 1. Start the Django development server

```bash
python manage.py runserver
```

### 2. Access the application

Open your browser and navigate to:
```
http://localhost:8000
```

### 3. Upload files

#### Option A: Upload Images
- Click or drag multiple images (JPG, PNG, JPEG)
- Upload at least front, rear, left, and right views for 360° coverage
- Can include interior views (dashboard, seats, etc.)

#### Option B: Upload Video
- Upload a single video file (MP4, AVI, MOV, MKV)
- Walk around the vehicle recording all angles
- System will automatically extract and classify frames

### 4. View results

The system will display:
- ✅ **360° Coverage Status**: Whether all required exterior views are present
- 📊 **Detected Views**: List of all detected views with confidence scores
- 📝 **Detailed Analysis**: AI-generated inspection report for each view
- ⚠️ **Missing Views**: List of views needed for complete 360° coverage

## 🔌 API Endpoints

### Check 360° Coverage

Analyze uploaded images or video and return detailed inspection report.

**Endpoint**: `/api/check360/`

**Method**: `POST`

**Content-Type**: `multipart/form-data`

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `files` | File[] or File | Multiple image files OR a single video file |

**Request Example (cURL)**:

```bash
# Upload images
curl -X POST http://localhost:8000/api/check360/ \
  -F "files=@front.jpg" \
  -F "files=@rear.jpg" \
  -F "files=@left.jpg" \
  -F "files=@right.jpg"

# Upload video
curl -X POST http://localhost:8000/api/check360/ \
  -F "files=@vehicle_360.mp4"
```

**Response Format**:

```json
{
  "is_complete_360": true,
  "detected_360_views": ["front", "rear", "left", "right"],
  "missing_360_views": [],
  "detected_views": ["front", "rear", "left", "right", "dashboard", "console"],
  "processing_mode": "video",
  "total_frames_processed": 45,
  "frames_classified": 28,
  "details": [
    {
      "frame": "frame_0023",
      "view": "front (95.32%)",
      "confidence": 95.32,
      "sharpness_score": 1234.56,
      "api_data": {
        "Brand": "Toyota",
        "Model": "Camry",
        "Year": "2022",
        "Colour": "Silver",
        "License_plate": "detected",
        "Headlight_status": "ok",
        "Grille_condition": "ok",
        "Bumper_condition": "minor scratch",
        "Windshield_condition": "ok",
        "Paint_quality": "good",
        "Emblem/logo_condition": "ok"
      },
      "image_path": "/media/360_processed_images/front/frame_0023.jpg"
    },
    {
      "frame": "frame_0089",
      "view": "dashboard (92.45%)",
      "confidence": 92.45,
      "sharpness_score": 1567.89,
      "api_data": {
        "Temperature": "normal",
        "Trip_meter": "1234.5 km",
        "Warning_lights": "none",
        "Fuel": "3/4 full",
        "Dashboard_condition": "excellent"
      },
      "image_path": "/media/360_processed_images/dashboard/frame_0089.jpg"
    }
  ]
}
```

**Response Fields**:
- `is_complete_360`: Boolean indicating if all 4 exterior views are detected
- `detected_360_views`: Array of exterior views found
- `missing_360_views`: Array of exterior views needed for 360° coverage
- `detected_views`: All views detected (exterior + interior)
- `processing_mode`: Either "images" or "video"
- `details`: Array of detailed analysis for each classified view

### Serve Processed Videos

**Endpoint**: `/media/videos/<filename>`

**Method**: `GET`

**Description**: Serves processed video files with proper content-type headers

**Example**:
```
GET http://localhost:8000/media/videos/processed_vehicle_360.mp4
```

## 🔍 How It Works

### 1. File Upload & Validation

The frontend (`templates/index.html`) validates:
- File types (images vs video)
- Prevents mixing images and videos in one upload
- Checks file size limits

```javascript
function handleFileSelection(files) {
    const validImageTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    const validVideoTypes = ['video/mp4', 'video/avi', 'video/mov'];
    // Validation logic ensures proper file types
}
```

### 2. Video Frame Extraction

**Function**: `extract_frames_from_video()`

- Opens video with OpenCV
- Extracts frames at specified intervals (default: every 3rd frame)
- Processes frames in batches to optimize memory
- Stores frames temporarily for classification
- Returns list of frame paths and temporary directory

**Memory Optimization**:
```python
skip_frames = 15  # Extract every 15th frame
batch_size = 10   # Process 10 frames at a time
```

### 3. View Classification with YOLO

**Function**: `classify_view()`

- Loads YOLO model trained on vehicle views
- Classifies each frame/image
- Returns:
  - Predicted class (view type)
  - Confidence score (0-100%)
  - Bounding box coordinates

**Confidence Threshold**: 85%

### 4. Smart Frame Selection

**Function**: `process_frames()`

For each view class, selects the best frame based on:

1. **Confidence Score**: YOLO classification confidence
2. **Sharpness Score**: Laplacian variance (higher = sharper)
3. **Combined Score**: `confidence × sharpness`

```python
combined_score = conf * calculate_sharpness(frame_path)
```

This ensures:
- High classification accuracy
- Clear, sharp images for analysis
- Best representative frame for each view

### 5. AI-Powered Detailed Analysis

**Function**: `analyze_frame()`

For each selected frame:

1. **Encodes** image to base64
2. **Sends** to Ollama API with view-specific prompt
3. **Extracts** structured information:
   - Vehicle details (brand, model, year, color)
   - Component conditions
   - Damage assessment
   - Warning indicators
4. **Returns** JSON-formatted analysis

**Prompt Engineering**: Each view type has a custom prompt:

```python
class_prompts = {
    "dashboard": {
        "System": "Analyze dashboard and extract temperature, trip meter, warning lights, fuel level...",
        "Keys": ["Temperature", "Trip_meter", "Warning_lights", "Fuel", "Dashboard_condition"]
    },
    "front": {
        "System": "Analyze front view and extract brand, model, license plate, headlight status...",
        "Keys": ["Brand", "Model", "Year", "Colour", "Paint_quality", "License_plate", ...]
    },
    # ... more prompts for each view
}
```

### 6. 360° Coverage Verification

**Required Exterior Views**: `{"front", "left", "right", "rear"}`

The system:
- Checks if all 4 exterior views are detected
- Identifies missing views
- Returns coverage status

### 7. Response Generation

Compiles all analysis into structured JSON response with:
- Coverage status
- Detected and missing views
- Detailed inspection data for each view
- Image paths for frontend display

## 📝 Valid View Types

### Exterior Views (Required for 360°):
| View | Description | Analysis Includes |
|------|-------------|-------------------|
| `front` | Front view of vehicle | Brand, model, license plate, headlights, grille, bumper, windshield |
| `rear` | Rear view of vehicle | License plate, taillights, bumper, trunk, rear windshield |
| `left` | Left side view | Doors, mirrors, windows, side condition |
| `right` | Right side view | Doors, mirrors, windows, side condition |

### Interior Views (Optional):
| View | Description | Analysis Includes |
|------|-------------|-------------------|
| `dashboard` | Dashboard and instrument cluster | Temperature, trip meter, warning lights, fuel level |
| `console` | Center console | Infotainment, climate control, air vents, features |
| `seats` | Seat conditions | Material, color, condition |
| `steering_wheel` | Steering wheel area | Type, condition |
| `gear_stick` | Gear shift area | Position, type (manual/automatic), condition |

## 🛠️ Technologies Used

### Backend
- **Django 4.2+**: Web framework
- **Django REST Framework**: API endpoints
- **Ultralytics YOLO (YOLOv8)**: Object detection and classification
- **OpenCV (cv2)**: Image and video processing
- **NumPy**: Numerical operations
- **Pillow**: Image manipulation

### AI/ML
- **YOLO**: Pre-trained model for vehicle view classification
- **Ollama API**: Large Language Model for detailed image analysis
- **Computer Vision**: Sharpness calculation using Laplacian variance

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling with responsive design
- **JavaScript**: Interactive file upload and form handling
- **Fetch API**: Asynchronous server communication

### Infrastructure
- **SQLite**: Database (development)
- **FFmpeg**: Video codec support
- **Python tempfile**: Temporary file management

## 🎯 Configuration Parameters

### Frame Extraction (Video Processing)

```python
skip_frames = 15        # Extract every 15th frame (reduce for more frames)
batch_size = 10         # Process 10 frames at a time (memory management)
```

### Classification Thresholds

```python
MIN_CONFIDENCE = 0.85   # 85% minimum confidence for classification
```

### API Configuration

```python
OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT = 30           # API request timeout in seconds
```

### File Upload Limits

In `car360/settings.py`:

```python
DATA_UPLOAD_MAX_MEMORY_SIZE = 104857600  # 100 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 104857600  # 100 MB
```

## 🐛 Debugging

### Enable Detailed Logging

In `detection/views.py`:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### Check YOLO Model

```python
python manage.py shell
```

```python
from ultralytics import YOLO
model = YOLO('model/best.pt')
print(model.names)  # Should show all class names
```

### Test Ollama Connection

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llava",
  "prompt": "Test",
  "stream": false
}'
```

## 🔒 Security Considerations

- ✅ **File Type Validation**: Only allowed file types can be uploaded
- ✅ **Path Traversal Protection**: Prevents directory traversal attacks
- ✅ **CSRF Protection**: Django CSRF tokens for POST requests
- ✅ **Temporary File Cleanup**: Automatic cleanup prevents disk space issues
- ✅ **Input Sanitization**: Filenames and paths are validated

**Production Recommendations**:
- Use PostgreSQL or MySQL instead of SQLite
- Configure proper `ALLOWED_HOSTS` in settings
- Enable HTTPS
- Use environment variables for sensitive data
- Implement rate limiting
- Add authentication/authorization

## 📊 Performance Optimization

### Current Optimizations

1. **Batch Processing**: Video frames processed in batches
2. **Memory Management**: Temporary files cleaned up immediately
3. **Frame Selection**: Only best frames sent to AI for analysis
4. **Lazy Loading**: Models loaded once at startup
5. **Efficient Storage**: Processed images organized by view type

### Performance Tips

- **Use GPU**: Install CUDA for 10-100x faster YOLO inference
- **Adjust skip_frames**: Higher values = faster processing, fewer frames
- **Reduce Video Resolution**: Pre-process videos to 720p or 1080p
- **Use SSD**: Store media files on SSD for faster I/O

## 🔧 Troubleshooting

### Common Issues

#### 1. Ollama Not Responding

**Error**: `Connection refused to localhost:11434`

**Solution**:
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# Pull a model if needed
ollama pull llava
```

#### 2. Model Not Found

**Error**: `FileNotFoundError: model/best.pt`

**Solution**:
- Verify `best.pt` exists in the `model/` directory
- Check file path in `detection/views.py`
- Ensure you have the trained YOLO model

#### 3. Video Processing Fails

**Error**: `Could not open video file`

**Solutions**:
- Install FFmpeg and add to PATH
- Check video file is not corrupted
- Verify video format is supported (MP4, AVI, MOV)
- Try converting video: `ffmpeg -i input.mov -c:v libx264 output.mp4`

#### 4. Low Confidence Detections

**Issue**: All frames showing < 85% confidence

**Solutions**:
- Ensure good lighting in images/videos
- Use higher resolution images (min 640x640)
- Check if views are clearly visible
- Consider retraining YOLO model with more data
- Adjust confidence threshold in code

#### 5. Out of Memory Error

**Error**: `CUDA out of memory` or `MemoryError`

**Solutions**:
- Reduce `batch_size` in `extract_frames_from_video()`
- Increase `skip_frames` to process fewer frames
- Use CPU instead of GPU (slower but uses less memory)
- Close other applications using GPU/RAM

#### 6. Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'cv2'`

**Solution**:
```bash
pip install -r requirements.txt
```

If specific packages fail:
```bash
pip install opencv-python ultralytics django djangorestframework pillow numpy
```

## 📚 Additional Resources

- [Django Documentation](https://docs.djangoproject.com/)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [OpenCV Documentation](https://docs.opencv.org/)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to all functions
- Update README if adding new features
- Test thoroughly before submitting PR
- Include clear commit messages

## 📄 License

This project is proprietary software. All rights reserved.

## 👥 Authors

- **MegatonREX** - *Initial work* - [GitHub](https://github.com/MegatonREX)

## 🙏 Acknowledgments

- Ultralytics team for YOLO
- Ollama team for the AI inference platform
- Django community for the excellent web framework
- OpenCV contributors

## 📞 Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/MegatonREX/Vehicle-Inspection-Bot/issues)
- Check existing issues for similar problems
- Provide detailed error messages and logs

---

**⚠️ Important Notes**:

1. This system requires a **trained YOLO model** (`best.pt`) for vehicle view classification
2. Ensure **Ollama** is running before starting the application
3. The system works best with **clear, well-lit images** of vehicles
4. For production use, implement proper **authentication and security** measures

---

Made with ❤️ for automated vehicle inspection
