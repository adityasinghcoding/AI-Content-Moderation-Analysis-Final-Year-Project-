document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form');
    const predictionResult = document.getElementById('predictionResult');

    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission

        // Display loading animation
        predictionResult.innerHTML = '<p class="text-lg">Loading...</p>';

        // Submit form data asynchronously
        const formData = new FormData(form);
        try {
            const response = await fetch('/submit_prediction', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            // Update frontend with prediction result
            predictionResult.innerHTML = `<p class="text-lg">Text: <span class="text-blue-500">${data.text}</span></p>
                                          <p class="text-lg">Toxicity Result: <span class="text-red-500">${data.toxicity_result}</span></p>`;
        } catch (error) {
            console.error('Error:', error);
            predictionResult.innerHTML = '<p class="text-lg text-red-500">An error occurred. Please try again later.</p>';
        }
    });
});
