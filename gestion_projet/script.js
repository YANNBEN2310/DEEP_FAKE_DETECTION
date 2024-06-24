function openModal(imgElement) {
    var modal = document.getElementById("modal");
    var modalImg = document.getElementById("modal-img");
    modal.style.display = "block";
    modalImg.src = imgElement.src;
}

function closeModal() {
    var modal = document.getElementById("modal");
    modal.style.display = "none";
}

function saveTable() {
    const table = document.querySelector(".ordonnancement-table tbody");
    const rows = Array.from(table.rows);
    const data = rows.map(row => {
        return Array.from(row.cells).map(cell => cell.textContent);
    });
    localStorage.setItem("tableData", JSON.stringify(data));
    alert("Table saved!");
}

function loadTable() {
    const table = document.querySelector(".ordonnancement-table tbody");
    const data = JSON.parse(localStorage.getItem("tableData"));
    if (data) {
        table.innerHTML = data.map(row => {
            return `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`;
        }).join('');
    }
}

document.addEventListener("DOMContentLoaded", loadTable);

const darkModeToggle = document.getElementById('dark-mode-toggle');
darkModeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
});
