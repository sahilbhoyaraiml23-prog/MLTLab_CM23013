let model;

async function loadModel() {
    document.getElementById("result").innerText = "Loading AI Model...";
    model = await mobilenet.load();
    document.getElementById("result").innerText = "Model Loaded! Upload an image.";
}

loadModel();

document.getElementById("imageUpload").addEventListener("change", function(event) {
    const file = event.target.files[0];
    if (!file) return;

    const img = document.getElementById("preview");
    img.src = URL.createObjectURL(file);

    document.getElementById("loader").style.display = "block";

    img.onload = async () => {
        const predictions = await model.classify(img);

        document.getElementById("loader").style.display = "none";

        document.getElementById("result").innerHTML =
            `🔍 <b>${predictions[0].className}</b><br>
             Confidence: ${(predictions[0].probability * 100).toFixed(2)}%`;
    };
});