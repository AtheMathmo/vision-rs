extern crate vision;

use vision::mnist::{MNISTBuilder};

fn main() {
    let builder = MNISTBuilder::new();
    let mnist = builder.data_home("MNIST")
                       .verbose()
                       .get_data().unwrap();
    println!("{}", mnist.train_imgs.len());
}