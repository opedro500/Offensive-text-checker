document.querySelector('#send').addEventListener('submit', (e) => {
    e.preventDefault();

    const text = document.querySelector('#verify_text').value;

    removeClass();

    if (!text) {
        document.querySelector('#error_comment').classList.add('show');
        return;
    }

    document.querySelector('#custom_loader_box').classList.add('show');

    const url = "http://localhost:8000/check";

    const data = JSON.stringify({
        text: text
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

            if (data.is_offensive) {
                document.querySelector('#bad_result_test').classList.add('show');
                console.log(`Ofensivo! Certeza do modelo: ${data.confidence}`);
            } else {
                document.querySelector('#good_result_test').classList.add('show');
                console.log(`Seguro! Certeza da modelo: ${data.confidence}`);
            }
        })
        .catch(error => {
            console.error("Erro ao analisar o texto:", error);
            removeClass();
        });
});

function removeClass() {
    document.querySelector('#custom_loader_box').classList.remove('show');
    document.querySelector('#error_comment').classList.remove('show');
    document.querySelector('#good_result_test').classList.remove('show');
    document.querySelector('#bad_result_test').classList.remove('show');
};