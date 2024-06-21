document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("load-media").addEventListener("change", function() {
        handleFileSelect(this);
    });
});

function handleFileSelect(input) {
    const fileMessage = document.getElementById('file-message');
    const fileTypeInput = document.getElementById('file-type');
    if (input.files && input.files[0]) {
        const fileName = input.files[0].name;
        fileMessage.textContent = `Selected file: ${fileName}`;
        const fileType = input.files[0].type.startsWith('image') ? 'image' : 'video';
        fileTypeInput.value = fileType;
        document.getElementById('upload-form').submit();
    } else {
        fileMessage.textContent = 'No file selected';
        fileTypeInput.value = '';
    }
}

function startProgressBar() {
    const progressBarInner = document.getElementById('progress-bar-inner');
    progressBarInner.style.width = '0%';
    progressBarInner.textContent = '0%';
    let width = 0;
    const interval = setInterval(function() {
        if (width >= 100) {
            clearInterval(interval);
        } else {
            width++;
            progressBarInner.style.width = width + '%';
            progressBarInner.textContent = width + '%';
        }
    }, 100);
}
