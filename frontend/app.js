document.addEventListener('DOMContentLoaded', function () {
    const checkBtn = document.getElementById('checkBtn');
    const messageInput = document.getElementById('message');
    const resultDiv = document.getElementById('result');

    checkBtn.addEventListener('click', () => {
        const message = messageInput.value;

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        })
        .then(response => response.json())
        .then(data => {
            const prediction = data.prediction;
            resultDiv.innerText = `Result: ${prediction}`;

            if (prediction.toLowerCase() === "spam") {
                messageInput.style.boxShadow = "0 0 12px 2px red";
            } else if (prediction.toLowerCase() === "not spam") {
                messageInput.style.boxShadow = "0 0 12px 2px limegreen";
            } else {
                console.log("error in getting the result");
                messageInput.style.boxShadow = "none";
            }

            // Reset the styles after 3 seconds
            setTimeout(() => {
                messageInput.style.boxShadow = "none";
                resultDiv.style.border = "none";
            }, 3000);
        })
        .catch(error => {
            console.error('Error:', error);
            resultDiv.innerText = 'Something went wrong!';
            messageInput.style.boxShadow = "0 0 12px 2px orange";
        });
    });
});
