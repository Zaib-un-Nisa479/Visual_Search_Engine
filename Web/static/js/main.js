// Main JavaScript for Ring Search Engine - FIXED VERSION

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const processingSection = document.getElementById('processingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const originalImage = document.getElementById('originalImage');
const processedImage = document.getElementById('processedImage');
const stoneRegion = document.getElementById('stoneRegion');
const qualityMetrics = document.getElementById('qualityMetrics');
const matchGrid = document.getElementById('matchGrid');
const ringBadge = document.getElementById('ringBadge');
const stoneBadge = document.getElementById('stoneBadge');
const stoneType = document.getElementById('stoneType');
const errorMessage = document.getElementById('errorMessage');

let uploadedFile = null;

// Upload Area Click Handler
if (uploadArea) {
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
}

// File Input Change Handler
if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

// Drag and Drop Handlers
if (uploadArea) {
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.transform = 'translateY(-5px)';
        uploadArea.style.boxShadow = '0 15px 40px rgba(0, 0, 0, 0.3)';
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.style.transform = 'translateY(0)';
        uploadArea.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.2)';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.transform = 'translateY(0)';
        uploadArea.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.2)';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
}

// Handle File Upload
function handleFileUpload(file) {
    console.log('📁 File selected:', file.name);
    
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file (JPG, PNG, JPEG)');
        return;
    }
    
    if (file.size > 16 * 1024 * 1024) {
        showError('File size exceeds 16MB limit');
        return;
    }
    
    uploadedFile = file;
    
    // Show preview in original image section
    const reader = new FileReader();
    reader.onload = (e) => {
        if (originalImage) {
            originalImage.innerHTML = `<img src="${e.target.result}" alt="Uploaded ring" style="max-width:100%; max-height:200px; border-radius:8px;">`;
        }
    };
    reader.readAsDataURL(file);
    
    // Process the image
    processImage();
}

// Process Image with Backend
async function processImage() {
    console.log('🔄 Processing image...');
    
    // Show processing section
    if (uploadArea) uploadArea.style.display = 'none';
    if (processingSection) processingSection.style.display = 'block';
    if (resultsSection) resultsSection.style.display = 'none';
    if (errorSection) errorSection.style.display = 'none';
    
    const formData = new FormData();
    formData.append('file', uploadedFile);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        console.log('📥 Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('📊 Response data:', data);
        
        if (data.success !== false) {
            displayResults(data);
        } else {
            showError(data.error || data.message || 'Failed to process image');
        }
        
    } catch (error) {
        console.error('❌ Error:', error);
        showError('Network error: ' + error.message);
    }
    
    if (processingSection) processingSection.style.display = 'none';
}

// Display Results
function displayResults(data) {
    console.log('🎨 Displaying results...');
    
    // Update badges
    if (ringBadge) {
        if (data.has_ring) {
            ringBadge.textContent = '✅ Ring Detected';
            ringBadge.className = 'badge success';
        } else {
            ringBadge.textContent = '❌ No Ring Detected';
            ringBadge.className = 'badge error';
        }
    }
    
    if (stoneBadge) {
        if (data.has_stone) {
            stoneBadge.textContent = '💎 Stone Detected';
            stoneBadge.className = 'badge success';
        } else {
            stoneBadge.textContent = 'No Stone Detected';
            stoneBadge.className = 'badge warning';
        }
    }
    
    // Display stone type
    if (stoneType && data.has_stone && data.stone_type && data.stone_type !== 'none') {
        stoneType.textContent = `Stone Type: ${data.stone_type.toUpperCase()}`;
        stoneType.style.display = 'block';
    } else if (stoneType) {
        stoneType.style.display = 'none';
    }
    
    // Display processed image
    if (processedImage) {
        if (data.processed_image) {
            console.log('🖼️ Setting processed image');
            processedImage.innerHTML = `<img src="${data.processed_image}" alt="Processed ring" style="max-width:100%; max-height:200px; border-radius:8px;">`;
        } else {
            processedImage.innerHTML = '<p style="color:#999;">No processed image available</p>';
        }
    }
    
    // Display stone region
    if (stoneRegion) {
        if (data.stone_region) {
            console.log('💎 Setting stone region');
            stoneRegion.innerHTML = `<img src="${data.stone_region}" alt="Stone region" style="max-width:100%; max-height:200px; border-radius:8px;">`;
        } else {
            stoneRegion.innerHTML = '<p style="color:#999;">No stone region detected</p>';
        }
    }
    
    // Display quality metrics
    if (qualityMetrics && data.quality_metrics) {
        let metricsHtml = '<div style="display:grid; gap:10px;">';
        for (const [key, value] of Object.entries(data.quality_metrics)) {
            metricsHtml += `<div><strong>${key}:</strong> ${value}</div>`;
        }
        metricsHtml += '</div>';
        qualityMetrics.innerHTML = metricsHtml;
    } else if (qualityMetrics) {
        qualityMetrics.innerHTML = '<p style="color:#999;">No metrics available</p>';
    }
    
    // Display matches
    if (matchGrid) {
        if (data.matches && data.matches.length > 0) {
            console.log(`🔍 Displaying ${data.matches.length} matches`);
            let matchesHtml = '';
            data.matches.forEach((match, index) => {
                const similarityPercent = Math.round(match.similarity * 100);
                const similarityColor = similarityPercent > 80 ? '#4caf50' : (similarityPercent > 60 ? '#ff9800' : '#f44336');
                
                matchesHtml += `
                    <div class="match-card" style="background:white; border-radius:12px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.1); transition:transform 0.2s; cursor:pointer;">
                        <div class="match-image" style="width:100%; height:180px; overflow:hidden; background:#f5f5f5;">
                            <img src="${match.image}" alt="${match.filename}" style="width:100%; height:100%; object-fit:cover;">
                        </div>
                        <div class="match-info" style="padding:12px;">
                            <div class="match-name" style="font-weight:600; margin-bottom:4px;">${match.name || match.filename}</div>
                            <div class="match-similarity" style="color:${similarityColor}; font-weight:500;">${similarityPercent}% Match</div>
                        </div>
                    </div>
                `;
            });
            matchGrid.innerHTML = matchesHtml;
        } else {
            console.log('No matches found');
            matchGrid.innerHTML = '<p style="text-align: center; grid-column: 1/-1; color:#999;">No similar rings found in catalog.</p>';
        }
    }
    
    // Show results section
    if (resultsSection) resultsSection.style.display = 'block';
    console.log('✅ Results displayed successfully');
}

// Show Error
function showError(message) {
    console.error('❌ Error:', message);
    
    if (errorMessage) errorMessage.textContent = message;
    if (errorSection) errorSection.style.display = 'block';
    if (resultsSection) resultsSection.style.display = 'none';
    
    // Reset upload area
    if (uploadArea) uploadArea.style.display = 'block';
    
    // Auto-hide error after 5 seconds
    setTimeout(() => {
        if (errorSection) errorSection.style.display = 'none';
    }, 5000);
}

// Reset Upload
function resetUpload() {
    console.log('🔄 Resetting upload');
    if (uploadArea) uploadArea.style.display = 'block';
    if (errorSection) errorSection.style.display = 'none';
    if (resultsSection) resultsSection.style.display = 'none';
    if (fileInput) fileInput.value = '';
    uploadedFile = null;
    
    // Clear images
    if (originalImage) originalImage.innerHTML = '<p style="color:#999;">No image loaded</p>';
    if (processedImage) processedImage.innerHTML = '<p style="color:#999;">Processing...</p>';
    if (stoneRegion) stoneRegion.innerHTML = '<p style="color:#999;">No stone detected</p>';
    if (matchGrid) matchGrid.innerHTML = '<p style="text-align: center; grid-column: 1/-1; color:#999;">No matches found</p>';
}

// Load Database Function
async function loadDatabase() {
    console.log('📚 Loading database...');
    try {
        const response = await fetch('/load_database', {
            method: 'POST'
        });
        const data = await response.json();
        console.log('Database response:', data);
        
        if (data.success) {
            alert(`Database loaded! Stone rings: ${data.stone_count || 0}, Plain rings: ${data.plain_count || 0}`);
        } else {
            alert('Failed to load database');
        }
    } catch (error) {
        console.error('Error loading database:', error);
        alert('Error loading database');
    }
}

// Update stats
async function updateStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        const statsDiv = document.getElementById('stats');
        if (statsDiv) {
            statsDiv.innerHTML = `📊 Catalog: ${data.catalog_count || 0} rings available`;
        }
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Page loaded, initializing...');
    updateStats();
});