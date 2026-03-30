"""
Configuration settings for the Ring Visual Search Engine
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CATALOG_DIR = os.path.join(DATA_DIR, 'catalog')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')

# Create directories if they don't exist
for dir_path in [DATA_DIR, CATALOG_DIR, PROCESSED_DIR, FEATURES_DIR, UPLOAD_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Model settings
IMAGE_SIZE = (224, 224)
FEATURE_EXTRACTOR = 'vgg16'
USE_PRETRAINED = True

# Classification thresholds
RING_CONFIDENCE_THRESHOLD = 0.6
STONE_CONFIDENCE_THRESHOLD = 0.65

# Processing settings
TARGET_SIZE = (512, 512)
BACKGROUND_COLOR = (255, 255, 255)  # White
DENOISE_STRENGTH = 10
SHARPEN_STRENGTH = 1.0

# Matching settings
TOP_N_MATCHES = 5
SIMILARITY_THRESHOLD = 0.5

# Web settings
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}