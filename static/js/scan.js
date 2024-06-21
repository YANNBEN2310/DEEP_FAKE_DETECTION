document.addEventListener("DOMContentLoaded", function() {
    var scanNowButton = document.querySelector(".button-scan-now");
    
    scanNowButton.addEventListener("click", function(event) {
        event.preventDefault();
        
        var imageInput = document.getElementById('load-image');
        var videoInput = document.getElementById('load-video');
        
        if (imageInput.files.length > 0) {
            imageInput.form.submit();
        } else if (videoInput.files.length > 0) {
            videoInput.form.submit();
        } else {
            alert("Please load an image or a video before scanning.");
        }
    });
});
