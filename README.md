![](https://static.wikia.nocookie.net/pokemon-daybreak/images/e/ef/470.png/revision/latest?cb=20200731231258)

# Leafeon 🥬🦮

Leafeon is a pure rust fully connected dense neural network implementation.

## Structure

-   `leafeon-core`: internal neural network implementation
-   `leafeon-types`: common types used by leafeon crates
-   `leafeon-gpu`: matrix multiplication and outer product GPU implementation. unused
-   `leafeon-server`: leafeon server (used with the ui)
-   `leafeon-ui`: react app that allows users to draw digits and get predictions

## Usage

```rust
/// MNIST digits example
let dataset = load_data()
	.labels_path("./data/train-labels-idx1-ubyte")
	.data_path("./data/train-images-idx3-ubyte")
	.call()?;

let preprocess = ();
let preprocess = RotateLayer::new(preprocess, std::f32::consts::PI * 0.1);
let preprocess = OffsetLayer::new(preprocess, 2.0);
let preprocess = ScaleLayer::new(preprocess, 0.1);
let preprocess = NoiseLayer::new(preprocess, 0.2);
let network = untrained().with_preprocessing(preprocess);
let network = network
	.train()
	.dataset(dataset)
	.accuracy(32.0 / 60_000.0)
	//.accuracy(1.0)
	.epochs(10)
	.learning_rate(0.005)
	//.learning_rate(1.0)
	.call();

tracing::info!("finished training");
network.save_data(save_path)?;
tracing::info!("succesfully saved train data");
```
