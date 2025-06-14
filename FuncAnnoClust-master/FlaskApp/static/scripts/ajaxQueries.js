const loaderImg = '<img width="50px" src="static/images/processing.gif">'

function loadScript(url, callback) {
    const script = document.createElement('script');
    script.src = url;
    script.async = true;
    script.onload = callback;
    document.body.appendChild(script);
}

loadScript('/static/scripts/requests.js', () => {
    console.log('Модуль запросов подключен');
})

function loader(elem) {
    elem.innerHTML = loaderImg;
}

document.getElementById('clsFormBtn').addEventListener('click', async () => {
    const formElem = document.getElementById('clsForm');
    const formData = new FormData(formElem);

    const resultsElem = document.getElementById('classificationResult');

    loader(resultsElem);

    const response = await request({
        method: 'POST',
        url: '/classify',
        data: formData
    })

    resultsElem.innerHTML = response;
    document.getElementById('resetBtn').style.display = 'block';
})


function reset() {
    document.getElementById('classificationResult').innerHTML = '';
}


$(function () {
    $('#countFormBtn').click(function () {
        const categoryElems = document.querySelectorAll('tr.category');
        const systemElems = document.querySelectorAll('tr.systems');

        let categories = {};

        for (let i = 0; i < categoryElems.length; i++) {
            const category = categoryElems[i];

            const categoryName = category.querySelector('td.category__name').innerText;
            const isSelected = category.querySelector('.category-btn.btn-secondary') !== null;

            const categorySistemElems = systemElems[i].querySelectorAll('input[type=checkbox]:checked');
            let categorySistems = [];
            for (const system of categorySistemElems) {
                categorySistems.push(system.value);
            }

            categories[categoryName] = {
                selected: isSelected,
                systems: categorySistems
            };
        }

        console.log('Запущен рассчет', categories);

        $('#countSlide').html(loaderImg);
        $.ajax({
            type: 'POST',
            url: '/count',
            data: JSON.stringify(categories),
            contentType: 'application/json',
            cache: false,
            processData: false,
            success: function (data) {
                $('#countSlide').html(data);
            },
        });
    });
});
$(function () {
    $('#vslFormBtn').click(async () => {
        const vslForm = document.getElementById('vslForm');
        const formData = new FormData(vslForm);
        console.debug('Построить график: ', formData);

        $('#vslSlide').html(loaderImg);

        const response = await request({
            method: 'POST',
            url: '/visualize',
            data: formData
        });

        $('#vslSlide').html(response);
    });
});
$(function () {
    $('#uploadBreakdownBtn').click(function () {
        var form_data = new FormData($('#uploadBreakdown')[0]);
        $('#breakdownDisplay').html('<img width="50px" src="static/images/processing.gif">');
        $.ajax({
            type: 'POST',
            url: '/uploadBreakdown',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function (data) {
                $('#breakdownDisplay').html(data);
            },
        });
    });
});


const exportFormatSelect = document.getElementById('exportFormat');
const exportPlotsBtn = document.getElementById('vslExportPlotsBtn');
exportFormatSelect.addEventListener('click', async () => {
    const exportFormat = exportFormatSelect.options[exportFormatSelect.selectedIndex].value;
    exportPlotsBtn.setAttribute('href', `/download/plots?export_format=${exportFormat}`);
});

document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const formElem = document.getElementById('analyzeForm');
    const preFormData = new FormData(document.getElementById('vslForm'));
    const formData = new FormData(formElem);

    for (const [key, value] of preFormData.entries()) {
        formData.append(key, value);
    }

    console.debug('Провести анализ: ', formData);

    const response = await request({
        method: 'POST',
        url: '/analyze',
        data: formData
    })
    document.getElementById('analyzeResult').innerHTML = response;
});