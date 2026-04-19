# 🔍 Ring Visual Search Engine

An AI-powered visual search engine that identifies and matches ring designs from images using advanced computer vision techniques.

## 🌟 Features

- **🔍 Ring Detection** - Automatically detects if an image contains a ring using Hough Circle Transform and contour analysis
- **💎 Stone Classification** - Identifies whether rings have stones and classifies stone types (diamond, ruby, sapphire, emerald, etc.)
- **🎨 Background Removal** - Removes image backgrounds and isolates rings for better analysis
- **🔄 Rotation-Invariant Matching** - Finds similar rings regardless of orientation using Hu moments and Fourier transforms
- **🌐 Web Interface** - User-friendly web interface for easy searching and browsing
- **📊 Quality Metrics** - Analyzes image quality including sharpness, contrast, and brightness
- **📁 Catalog Management** - Build and manage a database of ring images

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Zaib-un-Nisa479/Visual_Search_Engine.git
cd Visual_Search_Engine
Create and activate virtual environment

bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
Install dependencies

bash
pip install -r requirements.txt
Add catalog images
Place your ring images in the data/catalog/ folder:

text
data/catalog/
├── ring1.jpg
├── ring2.jpg
├── ring3.jpg
└── ...
Build the feature database

bash
python main.py --build-db
Run the web application

bash
python run_web.py
Open your browser
Navigate to http://localhost:5000

📖 Usage
Web Interface (Recommended)
Start the server

bash
python run_web.py
Upload an image

Click or drag & drop a ring image

Wait for processing

View analysis results and similar matches

Browse results

View ring detection status

Check stone detection

See quality metrics

Browse similar rings from catalog

Command Line Interface
bash
# Process a single image
python main.py --query path/to/ring.jpg

# Build database from catalog
python main.py --build-db

# Save results to file
python main.py --query ring.jpg --save results.json

# Start web server (alternative)
python main.py --web
📁 Project Structure
text
Visual_Search_Engine/
├── run_web.py              # Main web application entry point
├── main.py                 # CLI interface entry point
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
├── README.md              # This file
├── config/
│   └── settings.py        # Configuration settings
├── modules/               # Core modules
│   ├── classifier/        # Ring and stone classification
│   │   ├── ring_classifier.py
│   │   └── stone_classifier.py
│   ├── processor/         # Image processing
│   │   ├── ring_cropper.py
│   │   └── background_remover.py
│   └── matcher/           # Similarity matching
│       ├── stone_matcher.py
│       └── plain_matcher.py
├── Web/                   # Web interface
│   ├── templates/
│   │   └── index.html     # Main page
│   └── static/
│       ├── css/
│       │   └── style.css  # Styles
│       └── js/
│           └── main.js    # Frontend JavaScript
└── data/                  # Data directory
    ├── catalog/           # Catalog images
    ├── uploads/           # Temporary uploads
    ├── processed/         # Processed images
    └── features/          # Feature databases
🛠️ Technologies Used
Technology	Purpose
OpenCV	Computer vision and image processing
Flask	Web framework
NumPy	Numerical computations
Pillow	Image processing
scikit-image	Advanced image features (LBP, GLCM)
scikit-learn	Machine learning utilities
📊 How It Works
1. Ring Detection Pipeline
text
Upload Image → Ring Detection → Crop & Enhance → Background Removal → Stone Detection
2. Feature Extraction
Hu Moments - Shape descriptors (rotation invariant)

Local Binary Patterns - Texture analysis

GLCM Features - Texture patterns

Gabor Filters - Orientation analysis

3. Similarity Matching
Cosine similarity for feature vectors

Rotation invariance via circular shifts

Weighted combination of multiple features

🎯 Use Cases
Jewelry Stores - Find similar ring designs for customers

Online Retail - Visual search for ring catalogs

Design Inspiration - Find rings with similar patterns

Inventory Management - Categorize and organize ring collections

🔧 Configuration
Edit config/settings.py to customize:

python
# Model settings
IMAGE_SIZE = (224, 224)
FEATURE_EXTRACTOR = 'vgg16'

# Classification thresholds
RING_CONFIDENCE_THRESHOLD = 0.6
STONE_CONFIDENCE_THRESHOLD = 0.65

# Matching settings
TOP_N_MATCHES = 5
SIMILARITY_THRESHOLD = 0.5
