// Generate synthetic data: y = 2x + 1
const xs = tf.tensor1d([1, 2, 3, 4, 5]);
const ys = tf.tensor1d([3, 5, 7, 9, 11]);

// Create model
const model = tf.sequential();

// Dense layer
model.add(tf.layers.dense({
    units: 1,
    inputShape: [1]
}));

// Compile model
model.compile({
    optimizer: tf.train.sgd(0.1),
    loss: 'meanSquaredError'
});

// Train model
async function trainModel() {
    await model.fit(xs, ys, {
        epochs: 200,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if (epoch % 50 === 0) {
                    console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
                }
            }
        }
    });

    // Prediction
    const output = model.predict(tf.tensor2d([6], [1, 1]));
    output.print();
}

trainModel();
