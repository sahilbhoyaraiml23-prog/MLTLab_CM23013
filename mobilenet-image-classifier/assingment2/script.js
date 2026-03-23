const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let model;
let lastTime = performance.now();

// Start webcam
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true
    });

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
    console.log("Model loaded");
}

// Detection loop
async function detect() {
    const predictions = await model.classify(video);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "red";
    ctx.font = "20px Arial";

    ctx.fillText(predictions[0].className, 10, 30);

    let now = performance.now();
    let fps = 1000 / (now - lastTime);
    lastTime = now;

    document.getElementById("fps").innerText =
        "FPS: " + fps.toFixed(2);

    requestAnimationFrame(detect);
}

// Main
async function main() {
    await setupCamera();
    await loadModel();
    detect();
}

main();