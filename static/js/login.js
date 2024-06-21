document.addEventListener("DOMContentLoaded", function() {
    var loginBtn = document.getElementById("loginBtn");
    if (loginBtn) {
      loginBtn.addEventListener("click", function() {
        window.location.href = "{{ url_for('scan') }}";
      });
    }
  });