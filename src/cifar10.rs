use std::io;
use std::io::ErrorKind;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::fs::{create_dir_all, File};

use futures::{Future, Stream};
use hyper::Client;
use tokio_core::reactor::Core;

use flate2::read::GzDecoder;
use tar::Archive;

pub struct CIFAR10 {
    pub train_labels: Vec<u8>,
    pub train_imgs: Vec<Vec<u8>>,
    pub test_labels: Vec<u8>,
    pub test_imgs: Vec<Vec<u8>>
}

pub struct CIFAR10Builder {
    data_home: String,
    force_download: bool,
    verbose: bool
}

impl CIFAR10Builder {
    pub fn new() -> CIFAR10Builder {
        CIFAR10Builder {
            data_home: "CIFAR10".into(),
            force_download: false,
            verbose: false
        }
    }

    pub fn data_home<S: Into<String>>(mut self, dh: S) -> CIFAR10Builder {
        self.data_home = dh.into();
        self
    }

    pub fn force_download(mut self) -> CIFAR10Builder {
        self.force_download = true;
        self

    }

    pub fn verbose(mut self) -> CIFAR10Builder {
        self.verbose = true;
        self
    }

    pub fn get_data(self) -> io::Result<CIFAR10> {
        if self.verbose {
            println!("Creating data directory: {}", self.data_home);
        }
        create_dir_all(&self.data_home)?;

        if self.redownload() {
            if self.verbose { println!("Downloading CIFAR-10 data"); }
            self.download();
        } else if self.verbose { println!("Already downloaded"); }

        if self.verbose { println!("Extracting data"); }
        
        let (train_labels, train_imgs) = self.load_train_data()?;
        let (test_labels, test_imgs) = self.load_test_data()?;
        if self.verbose { println!("CIFAR-10 Loaded!"); }
        Ok(CIFAR10 {
            train_imgs: train_imgs,
            train_labels: train_labels,
            test_imgs: test_imgs,
            test_labels: test_labels
        })
    }

    /// Check whether dataset must be downloaded again
    fn redownload(&self) -> bool {
        if self.force_download {
            true
        } else {
            let file_names = [
                "cifar-10-batches-bin/data_batch_1.bin",
                "cifar-10-batches-bin/data_batch_2.bin",
                "cifar-10-batches-bin/data_batch_3.bin",
                "cifar-10-batches-bin/data_batch_4.bin",
                "cifar-10-batches-bin/data_batch_5.bin",
                "cifar-10-batches-bin/test_batch.bin"
            ];

            !file_names.iter().all(|f| self.get_file_path(f).is_file())
        }
    }

    fn get_file_path<P: AsRef<Path>>(&self, filename: P) -> PathBuf {
        Path::new(&self.data_home).join(filename)
    }

    fn download(&self) {
        let uri = String::from("http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz").parse().unwrap();

        let mut core = Core::new().unwrap();
        let client = Client::new(&core.handle());

        let get_data = client.get(uri).and_then(|res| {
            res.body().concat2()
        });
        let all_data = core.run(get_data).unwrap();
        let mut archive = Archive::new(GzDecoder::new(&*all_data));
        archive.unpack(self.data_home.clone()).unwrap();
    }

    fn load_train_data(&self)
        -> io::Result<(Vec<u8>, Vec<Vec<u8>>)>
    {
        let file_names = [
            "cifar-10-batches-bin/data_batch_1.bin",
            "cifar-10-batches-bin/data_batch_2.bin",
            "cifar-10-batches-bin/data_batch_3.bin",
            "cifar-10-batches-bin/data_batch_4.bin",
            "cifar-10-batches-bin/data_batch_5.bin"
        ];

        let mut train_labels = Vec::with_capacity(10000);
        let mut train_imgs = Vec::with_capacity(10000);

        for f in file_names.iter() {
            let full_path = self.get_file_path(f);
            let (mut batch_train_lbls, mut batch_train_data) = self.load_batch_file(full_path)?;
            train_labels.extend_from_slice(&mut batch_train_lbls);
            train_imgs.extend_from_slice(&mut batch_train_data)
        }

        Ok((train_labels, train_imgs))
    }

    fn load_test_data(&self)
        -> io::Result<(Vec<u8>, Vec<Vec<u8>>)>
    {
        let file_path = "cifar-10-batches-bin/test_batch.bin";
        let full_path = self.get_file_path(file_path);
        self.load_batch_file(full_path)
    }

    fn load_batch_file<P: AsRef<Path>>(&self, path: P)
        -> io::Result<(Vec<u8>, Vec<Vec<u8>>)>
    {
        let mut buf = vec![0u8; 3073];
        let mut file = File::open(path)?;

        let mut labels = vec![];
        let mut pixels = vec![];

        loop {
            match file.read_exact(&mut buf) {
                Ok(_) => {
                    labels.push(buf[0]);
                    pixels.push(buf[1..].into());
                },
                Err(e) => match e.kind() {
                    ErrorKind::UnexpectedEof => break,
                    _ => return Err(e)
                }
            }
        }

        Ok((labels, pixels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::remove_dir_all;

    #[test]
    #[ignore]
    fn test_builder() {
        let builder = CIFAR10Builder::new();
        let cifar10 = builder.data_home("CIFAR10").get_data().unwrap();
        assert_eq!(cifar10.train_imgs.len(), 50000);
        assert_eq!(cifar10.train_imgs[0].len(), 3072);
        assert_eq!(cifar10.train_labels.len(), 50000);
        assert_eq!(cifar10.test_imgs.len(), 10000);
        assert_eq!(cifar10.test_imgs[0].len(), 3072);
        assert_eq!(cifar10.test_labels.len(), 10000);
        remove_dir_all("CIFAR10").unwrap();
    }

}
