function handleFileSelect(input) {
    var file = input.files[0];
    var fileType = file.type;
    var framesInput = document.getElementById('frames-input');

    if (fileType.startsWith('video/')) {
      framesInput.style.display = 'block';
    } else {
      framesInput.style.display = 'none';
    }
  }