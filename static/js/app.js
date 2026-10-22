// Facial Expression Classification - Frontend JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const uploadStatus = document.getElementById('uploadStatus');
    const previewSection = document.getElementById('previewSection');
    const imagePreview = document.getElementById('imagePreview');
    const resultsSection = document.getElementById('resultsSection');
    const resultsList = document.getElementById('resultsList');
    const noResults = document.getElementById('noResults');
    const videoFeed = document.getElementById('videoFeed');
    const liveTabButton = document.getElementById('live-tab');
    
    // Only start the webcam stream when the live tab is opened
    if (videoFeed && liveTabButton) {
        liveTabButton.addEventListener('shown.bs.tab', function() {
            if (!videoFeed.getAttribute('src')) {
                videoFeed.src = videoFeed.dataset.src;
            }
        });
        liveTabButton.addEventListener('hidden.bs.tab', function() {
            videoFeed.removeAttribute('src');
        });
    }
    
    // Preview image when file selected
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewSection.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            showStatus('Please select a file first', 'danger');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        
        // Show loading status
        showStatus('<i class="fas fa-spinner fa-spin"></i> Processing image...', 'info');
        resultsSection.style.display = 'none';
        noResults.style.display = 'block';
        
        // Submit form
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showStatus('<i class="fas fa-check-circle"></i> Classification completed!', 'success');
                displayResults(data.results);
            } else {
                showStatus('<i class="fas fa-exclamation-circle"></i> ' + data.error, 'danger');
            }
        })
        .catch(error => {
            showStatus('<i class="fas fa-exclamation-triangle"></i> Error: ' + error.message, 'danger');
            console.error('Error:', error);
        });
    });
    
    function showStatus(message, type) {
        uploadStatus.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
    }
    
    function displayResults(results) {
        resultsList.innerHTML = '';
        noResults.style.display = 'none';
        resultsSection.style.display = 'block';
        
        results.forEach((result, index) => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'result-item';
            
            let html = `<h6><i class="fas fa-user"></i> Face ${index + 1}</h6>`;
            html += `<p class="mb-2"><strong>Predicted Emotion:</strong> <span class="badge bg-primary">${result.emotion}</span></p>`;
            html += `<p class="mb-3"><strong>Confidence:</strong> <span class="badge bg-success">${result.confidence}%</span></p>`;
            html += `<p class="mb-2"><strong>All Probabilities:</strong></p>`;
            
            // Sort probabilities
            const sortedProbs = Object.entries(result.probabilities)
                .sort((a, b) => b[1] - a[1]);
            
            sortedProbs.forEach(([emotion, prob]) => {
                html += `
                    <div class="prob-label">
                        <span>${emotion}</span>
                        <span>${prob.toFixed(1)}%</span>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: ${prob}%">
                            ${prob > 15 ? prob.toFixed(1) + '%' : ''}
                        </div>
                    </div>
                `;
            });
            
            resultDiv.innerHTML = html;
            resultsList.appendChild(resultDiv);
        });
    }
});
