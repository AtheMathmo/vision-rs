# Vision

This library provides access to common machine learning benchmarking datasets.

The library currently includes:

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)


Things are currently _very_ basic.

## Usage

Each dataset can be downloaded and processed using a Builder class. The builder is customizable in each case.

```rust
extern crate vision;

use vision::mnist::{MNISTBuilder};

fn main() {
    let builder = MNISTBuilder::new();
    let mnist = builder.data_home("MNIST")
                       .verbose()
                       .get_data().unwrap();
    println!("{}", mnist.train_imgs.len());
}
```

The MNIST object returned by the builder contains four public fields, `train_imgs`, `train_labels`, `test_images` and `test_labels`. The label fields are `Vec<u8>` types and the images are `Vec<Vec<u8>>`, each entry in the outermost `Vec` corresponds to a single datapoint.


Further preprocessing should be carried out by the user.
