document.querySelector('#toggle_model_btn').addEventListener('click', () => {
    document.querySelector('#select_wrapper').classList.toggle('show-select');
});

document.querySelector('#send').addEventListener('submit', (e) => {
    e.preventDefault();

    const text = document.querySelector('#verify_text').value;
    const level = document.querySelector('#model_level').value;

    removeClass();

    if (!text.trim()) {
        document.querySelector('#error_comment').classList.add('show');
        return;
    }

    document.querySelector('#custom_loader_box').classList.add('show');

    const url = "/predict";

    const data = JSON.stringify({
        text: text,
        level: level
    });

    fetch(url, {
        method: "POST",
        body: data,
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(response => response.json())
    .then(data => {
        removeClass();

        if (data.offensive) {
            const badBox = document.querySelector('#bad_result_test');
            badBox.querySelector('span').innerText = `Ofensivo (${data.confidence}%)`;
            badBox.classList.add('show');
        } else {
            const goodBox = document.querySelector('#good_result_test');
            const safeConfidence = (100 - data.confidence).toFixed(2);
            goodBox.querySelector('span').innerText = `Legal (${safeConfidence}%)`;
            goodBox.classList.add('show');
        }
    })
    .catch(error => {
        console.error("Erro ao analisar o texto:", error);
        removeClass();
        const errorBox = document.querySelector('#error_comment');
        errorBox.innerText = "Erro no servidor.";
        errorBox.classList.add('show');
    });
});

function removeClass() {
    document.querySelector('#custom_loader_box').classList.remove('show');
    document.querySelector('#error_comment').classList.remove('show');
    document.querySelector('#good_result_test').classList.remove('show');
    document.querySelector('#bad_result_test').classList.remove('show');
    
    document.querySelector('#error_comment').innerText = "Digite um comentário...";
}