let model = null;
let trainingData = [];
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');

// LOGGING
function log(msg) {
    console.log(msg);
    document.getElementById('logs').textContent += msg + '\n';
    document.getElementById('logs').scrollTop = 9999;
}

// Generate Synthetic Data: y = 2x + noise
function generateData(n = 100) {
    trainingData = [];
    for(let i = 0; i < n; i++) {
        const x = (Math.random() - 0.5) * 10;  // x: -5 to 5
        const y = 2 * x + (Math.random() - 0.5) * 1;  // y = 2x + noise
        trainingData.push({x, y});
    }
    log(`📊 Generated ${n} samples: y ≈ 2x + noise`);
    return {
        xs: tf.tensor2d(trainingData.map(d => [d.x])),
        ys: tf.tensor2d(trainingData.map(d => [d.y]))
    };
}

// 1. TRAIN MODEL (tf.sequential + tf.layers.dense)
async function trainModel() {
    log('🚀 Training Linear Regression...');
    document.getElementById('status').textContent = 'Training...';
    document.getElementById('status').className = 'status';
    
    const { xs, ys } = generateData(100);
    
    // Simple Linear Regression Model
    model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [1], units: 1 })  // y = wx + b
        ]
    });
    
    model.compile({
        optimizer: 'sgd',
        loss: 'meanSquaredError'
    });
    
    // Train
    const history = await model.fit(xs, ys, {
        epochs: 200,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if(epoch % 50 === 0) {
                    log(`Epoch ${epoch}: Loss = ${logs.loss.toFixed(4)}`);
                }
            }
        }
    });
    
    xs.dispose();
    ys.dispose();
    
    log('✅ Training Complete!');
    document.getElementById('status').textContent = 'Model Trained!';
    document.getElementById('status').className = 'status success';
    
    // Extract learned parameters
    const weights = model.getWeights()[0].dataSync()[0];
    const bias = model.getWeights()[1].dataSync()[0];
    log(`📐 Learned: y = ${weights.toFixed(2)}x + ${bias.toFixed(2)}`);
    
    drawChart();
}

// 2. SAVE MODEL
async function saveModel() {
    if(!model) return alert('Train first!');
    await model.save('localstorage://linear-regression');
    log('💾 Model saved locally!');
    document.getElementById('status').textContent = 'Model Saved!';
}

// 3. LOAD MODEL
async function loadModel() {
    try {
        model = await tf.loadLayersModel('localstorage://linear-regression');
        log('📂 Model loaded!');
        document.getElementById('status').textContent = 'Model Loaded!';
        document.getElementById('status').className = 'status success';
        drawChart();
    } catch(e) {
        alert('No saved model. Train first!');
    }
}

// 4. PREDICT
async function predict() {
    if(!model) return alert('Load/Train model first!');
    
    const x = parseFloat(document.getElementById('inputX').value);
    const input = tf.tensor2d([[x]]);
    const y = model.predict(input).dataSync()[0];
    
    document.getElementById('prediction').innerHTML = `
        <strong>y = ${y.toFixed(2)}</strong><br>
        Input x = ${x.toFixed(2)}
    `;
    
    log(`🔮 Prediction: x=${x.toFixed(2)} → y=${y.toFixed(2)}`);
    input.dispose();
    drawChart();
}

// VISUALIZATION
function drawChart() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1;
    for(let i = 0; i < 11; i++) {
        ctx.beginPath();
        ctx.moveTo(i*80, 0); ctx.lineTo(i*80, 400); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i*40); ctx.lineTo(800, i*40); ctx.stroke();
    }
    
    // Data points (blue)
    ctx.fillStyle = 'rgba(33,150,243,0.8)';
    trainingData.forEach(point => {
        const px = (point.x + 5) * 80;
        const py = 400 - (point.y + 10) * 20;
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI*2);
        ctx.fill();
    });
    
    // Learned line (if model exists)
    if(model) {
        ctx.strokeStyle = 'rgba(255,193,7,1)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        for(let px = 0; px < 800; px += 20) {
            const x = (px / 80) - 5;
            const input = tf.tensor2d([[x]]);
            const y = model.predict(input).dataSync()[0];
            const py = 400 - (y + 10) * 20;
            
            if(px === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
            input.dispose();
        }
        ctx.stroke();
    }
    
    // Axes
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, 400); ctx.lineTo(800, 400); ctx.lineTo(800, 0);
    ctx.stroke();
    
    ctx.fillStyle = 'white';
    ctx.font = '16px Arial';
    ctx.fillText('x (-5 to 5)', 750, 390);
    ctx.fillText('y', 10, 20);
}

// Initialize
log('✅ TensorFlow.js loaded');
log('🎯 Model: y = wx + b using tf.sequential() + tf.layers.dense()');
drawChart();