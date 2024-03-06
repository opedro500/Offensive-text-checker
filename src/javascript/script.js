document.querySelector('#send').addEventListener('submit', (e) => {
    e.preventDefault();

    const text = document.querySelector('#verify_text').value;

    if (!text) {
        removeClass();
        document.querySelector('#error_comment').classList.add('show');

        return;
    }

    const key = "AIzaSyBPEowkUcyUHjH8NT7mW1OAnDW3uGMH7SQ";
    const url = `https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key=${key}`;

    const data = JSON.stringify({
        comment: { text: text },
        languages: ["pt"],
        requestedAttributes: { TOXICITY: {} }
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

            if (data.attributeScores && data.attributeScores.TOXICITY.summaryScore.value > 0.5) {
                document.querySelector('#bad_result_test').classList.add('show');
            } else {
                document.querySelector('#good_result_test').classList.add('show');
            }
        })
        .catch(error => {
            console.error("Erro ao analisar o texto:", error);
        });
});

function removeClass() {
    document.querySelector('#error_comment').classList.remove('show');
    document.querySelector('#good_result_test').classList.remove('show');
    document.querySelector('#bad_result_test').classList.remove('show');
};