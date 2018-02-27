extern crate futures;
extern crate tokio_core;
extern crate hyper;
extern crate flate2;
extern crate tar;
extern crate byteorder;

pub mod mnist;
pub mod fashion_mnist;
pub mod cifar10;
pub mod cifar100;

#[cfg(test)]
mod tests {

}
