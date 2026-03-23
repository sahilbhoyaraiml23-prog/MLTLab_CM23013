const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let model;

// Start webcam
async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    return new Promise(resolve => {
        video.onloadedmetadata = () => {
            video.play();
            resolve(video);
        };
    });
}

// Load model
async function loadModel() {
    model = await mobilenet.load();
    console.log("Model Loaded");
}

// Detect objects
async function detect() {
    const predictions = await model.classify(video);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "red";
    ctx.font = "18px Arial";

    ctx.fillText(predictions[0].className, 10, 25);

    document.getElementById("result").innerText =
        "Prediction: " + predictions[0].className;

    requestAnimationFrame(detect);
}

// Main
async function main() {
    await startCamera();
    await loadModel();
    detect();
}

main();