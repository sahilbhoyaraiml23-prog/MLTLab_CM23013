let model;

async function loadModel() {
    document.getElementById("result").innerText = "Loading model...";
    model = await mobilenet.load();
    document.getElementById("result").innerText = "Model loaded! Upload image.";
}

loadModel();

document.getElementById("upload").addEventListener("change", function(event) {
    const file = event.target.files[0];
    if (!file) return;

    const img = document.getElementById("image");
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
        const predictions = await model.classify(img);

        document.getElementById("result").innerText =
            "Prediction: " + predictions[0].className +
            " (" + (predictions[0].probability * 100).toFixed(2) + "%)";
    };
});