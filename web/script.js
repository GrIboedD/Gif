// Функция для переключения между основными вкладками
document.querySelectorAll('.tab-link').forEach(link => {
    link.addEventListener('click', function(event) {
        event.preventDefault();

        // Удаляем класс active у всех ссылок и скрываем все содержимое
        document.querySelectorAll('.tab-link').forEach(link => link.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.style.display = 'none');

        const secondTab = document.querySelector('.tab-link[href="#content-2"]');
        const thirdTab = document.querySelector('.tab-link[href="#content-3"]');

        if (this.className === 'tab-link next-1-2') {
            secondTab.classList.add('active');
        } else if (this.className === 'tab-link next-3-1'){
            thirdTab.classList.add('active');
        } else {
            this.classList.add('active');
        }

        const targetId = this.getAttribute('href');
        document.querySelector(targetId).style.display = 'block';

        // Если переключаемся на первую вкладку, показываем под-вкладки
        if (targetId === '#content-1') {
            document.querySelectorAll('.sub-tab-content').forEach(content => content.style.display = 'none');
            document.querySelector('#content-1-1').style.display = 'block'; // Показать первую под-вкладку
            document.querySelector('.sub-tab-link.active').classList.remove('active'); // Деактивировать текущую под-вкладку
            document.querySelector('.sub-tab-link[href="#content-1-1"]').classList.add('active'); // Активировать первую под-вкладку
        }
        // Если переключаемся на вторую вкладку, показываем под-вкладки
        if (targetId === '#content-2') {
            document.querySelectorAll('.sub-tab-content').forEach(content => content.style.display = 'none');
            document.querySelector('#content-2-1').style.display = 'block'; // Показать первую под-вкладку
            document.querySelector('.sub-tab-link.active').classList.remove('active'); // Деактивировать текущую под-вкладку
            document.querySelector('.sub-tab-link[href="#content-2-1"]').classList.add('active'); // Активировать первую под-вкладку
        }

        // Если переключаемся на третью вкладку, показываем под-вкладки
        if (targetId === '#content-3') {
            document.querySelectorAll('.sub-tab-content').forEach(content => content.style.display = 'none');
            document.querySelector('#content-3-1').style.display = 'block'; // Показать первую под-вкладку
            document.querySelector('.sub-tab-link.active').classList.remove('active'); // Деактивировать текущую под-вкладку
            document.querySelector('.sub-tab-link[href="#content-3-1"]').classList.add('active'); // Активировать первую под-вкладку
        }
    });
});

// Функция для переключения между под-вкладками
document.querySelectorAll('.sub-tab-link').forEach(link => {
    link.addEventListener('click', function(event) {
        event.preventDefault();

        // Удаляем класс active у всех под-вкладок и скрываем все содержимое под-вкладок
        document.querySelectorAll('.sub-tab-link').forEach(link => link.classList.remove('active'));
        document.querySelectorAll('.sub-tab-content').forEach(content => content.style.display = 'none');

        const secondSecondTab = document.querySelector('.sub-tab-link[href="#content-2-2"]');

        if (this.className === 'sub-tab-link next-2-2') {
            secondSecondTab.classList.add('active');
            // alert(111)
            // alert(this.className)
        } else {
            this.classList.add('active');
            // alert(222)
            // alert(this.className)
        }

        const targetId = this.getAttribute('href');
        document.querySelector(targetId).style.display = 'block';

        // Если переключаемся на первую под-вкладку, показываем дополнительные вкладки
        if (targetId === '#content-1-2') {
            document.querySelectorAll('.sub-sub-tab-content').forEach(content => content.style.display = 'none');
            document.querySelector('#content-1-2-1').style.display = 'block'; // Показать первую доп. вкладку
            document.querySelector('.sub-sub-tab-link.active').classList.remove('active'); // Деактивировать текущую доп. вкладку
            document.querySelector('.sub-sub-tab-link[href="#content-1-2-1"]').classList.add('active'); // Активировать первую доп. вкладку
        }
    });
});

// Функция для переключения между дополнительными под-вкладками
document.querySelectorAll('.sub-sub-tab-link').forEach(link => {
    link.addEventListener('click', function(event) {
        event.preventDefault();

        // Удаляем класс active у всех дополнительных под-вкладок и скрываем все содержимое
        document.querySelectorAll('.sub-sub-tab-link').forEach(link => link.classList.remove('active'));
        document.querySelectorAll('.sub-sub-tab-content').forEach(content => content.style.display = 'none');

        // Добавляем класс active к текущей доп. вкладке и показываем соответствующее содержимое
        this.classList.add('active');
        const targetId = this.getAttribute('href');
        document.querySelector(targetId).style.display = 'block';
    });
});

function createTable() {
    const rowCount = document.getElementById("rowCount").value;
    const tableBody = document.getElementById("table-1").getElementsByTagName("tbody")[0];

    // Очищаем существующие строки
    tableBody.innerHTML = "";

    // Создаем новые строки
    for (let i = 0; i < rowCount; i++) {
        const row = document.createElement("tr");

        for (let j = 0; j < 4; j++) { // 4 колонки
            const cell = document.createElement("td");
            cell.textContent = `Строка ${i + 1}, Колонка ${j + 1}`;
            row.appendChild(cell);
        }

        tableBody.appendChild(row);
    }
}

document.getElementById('radio-btn').addEventListener('submit', function(event) {
    event.preventDefault(); // Предотвращаем стандартное поведение формы

    const selectedOption = document.querySelector('input[name="options"]:checked').value;
    alert('Вы выбрали: ' + selectedOption);
});

document.getElementById('btn-add-row').addEventListener('click', addRow);

function addRow() {
    // Получаем тело таблицы
    const tbody = document.getElementById('table-2').querySelector('tbody');

    // Создаем новую строку
    const newRow = document.createElement('tr');

    // Создаем 6 ячеек для новой строки
    for (let i = 1; i <= 6; i++) {
        const newCell = document.createElement('td');
        newCell.textContent = `Ячейка ${tbody.rows.length + 1}.${i}`; // Уникальный текст для каждой ячейки
        newRow.appendChild(newCell);
    }

    // Добавляем новую строку в тело таблицы
    tbody.appendChild(newRow);
}

// Функция для удаления всех строк таблицы, кроме заголовков
function deleteRows(tableId) {
    const tableBody = document.getElementById(tableId).getElementsByTagName("tbody")[0];

    // Проверка на наличие tbody
    if (!tableBody) {
        console.error(`Тело таблицы с ID "${tableId}" не найдено.`);
        return;
    }

    // Удаляем все строки из tbody
    while (tableBody.rows.length > 0) {
        tableBody.deleteRow(0);
    }
}

function getSelectedValue() {
    let select = document.getElementById("algorithm");
    let selectedValue = select.value; // Получаем выбранное значение

    let seedGroup = document.getElementById("seedGroup");
    let stopGroup = document.getElementById("stopGroup");
    let packageGroup = document.getElementById("packageGroup");

    seedGroup.style.visibility = 'hidden';
    stopGroup.style.visibility = 'hidden';
    packageGroup.style.visibility = 'hidden';

    if (selectedValue === "2" || selectedValue === "3") {
        seedGroup.style.visibility = 'visible';
        stopGroup.style.visibility = 'visible';
    }
    if (selectedValue === "3") {
        packageGroup.style.visibility = 'visible';
    }
}

function deleteOneRow(element) {
    // Получаем родительский элемент строки (tr)
    const row = element.closest('tr'); // Находим ближайший родительский элемент tr
    if (row) {
        row.remove(); // Удаляем строку (tr)
    }
}