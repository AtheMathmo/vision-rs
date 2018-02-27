use std::io;
use std::io::{Error, ErrorKind};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::fs::{create_dir_all, File};

use futures::{Future, Stream};
use futures::future::join_all;
use hyper::Client;
use tokio_core::reactor::Core;

use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;

const LABEL_MAGIC_NO: u32 = 2049;
const IMG_MAGIC_NO: u32 = 2051;

pub struct MNIST {
    pub train_labels: Vec<u8>,
    pub train_imgs: Vec<Vec<u8>>,
    pub test_labels: Vec<u8>,
    pub test_imgs: Vec<Vec<u8>>
}

pub struct MNISTBuilder {
    data_home: String,
    force_download: bool,
    verbose: bool
}

impl MNISTBuilder {
    pub fn new() -> MNISTBuilder {
        MNISTBuilder {
            data_home: "MNIST".into(),
            force_download: false,
            verbose: false
        }
    }

    pub fn data_home<S: Into<String>>(mut self, dh: S) -> MNISTBuilder {
        self.data_home = dh.into();
        self
    }

    pub fn force_download(mut self) -> MNISTBuilder {
        self.force_download = true;
        self

    }

    pub fn verbose(mut self) -> MNISTBuilder {
        self.verbose = true;
        self
    }

    pub fn get_data(self) -> io::Result<MNIST> {
        if self.verbose {
            println!("Creating data directory: {}", self.data_home);
        }
        create_dir_all(&self.data_home)?;

        if self.redownload() {
            if self.verbose { println!("Downloading MNIST data"); }
            self.download();
        } else if self.verbose { println!("Already downloaded"); }

        if self.verbose { println!("Extracting data"); }
        let (_train_lbl_meta, train_labels) = self.extract_labels(
            self.get_file_path("train-labels-idx1-ubyte.gz"))?;
        let (_train_img_meta, train_imgs) = self.extract_images(
            self.get_file_path("train-images-idx3-ubyte.gz"))?;
        let (_test_lbl_meta, test_labels) = self.extract_labels(
            self.get_file_path("t10k-labels-idx1-ubyte.gz"))?;
        let (_test_img_meta, test_imgs) = self.extract_images(
            self.get_file_path("t10k-images-idx3-ubyte.gz"))?;
        
        if self.verbose { println!("MNIST Loaded!"); }
        Ok(MNIST {
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
                "train-labels-idx1-ubyte.gz",
                "train-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz"
            ];

            !file_names.iter().all(|f| self.get_file_path(f).is_file())
        }
    }

    fn get_file_path(&self, filename: &str) -> PathBuf {
        Path::new(&self.data_home).join(filename)
    }

    fn download(&self) {
        let base_uri = String::from("http://yann.lecun.com/exdb/mnist/");
        let file_names = [
            "train-labels-idx1-ubyte.gz",
            "train-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz"
        ];


        let mut core = Core::new().unwrap();
        let client = Client::new(&core.handle());

        let all_gets = join_all(file_names.iter().map(move |f| {
            let full_uri = (base_uri.clone() + f).parse().unwrap();
            client.get(full_uri).and_then(move |res| {
                let path = self.get_file_path(f);
                let mut file = File::create(path).unwrap();
                res.body().for_each(move |chunk| {
                    file.write_all(&chunk)
                        .map(|_| ())
                        .map_err(From::from)
                })
            })
        }));
        core.run(all_gets).unwrap();
    }

    fn extract_labels<P: AsRef<Path>>(&self, label_file_path: P)
        -> io::Result<([u32; 2], Vec<u8>)>
    {
        let mut decoder = self.get_decoder(label_file_path)?;
        let mut metadata_buf = [0u32; 2];

        decoder.read_u32_into::<BigEndian>(&mut metadata_buf)?;

        let mut labels = Vec::new();
        decoder.read_to_end(&mut labels)?;
        if metadata_buf[0] != LABEL_MAGIC_NO {
            Err(Error::new(ErrorKind::InvalidData,
                "Unable to verify MNIST data. Force redownload."))
        } else {
            Ok((metadata_buf, labels))
        }
    }

    fn extract_images<P: AsRef<Path>>(&self, img_file_path: P)
        -> io::Result<([u32; 4], Vec<Vec<u8>>)>
    {
        let mut decoder = self.get_decoder(img_file_path)?;
        let mut metadata_buf = [0u32; 4];

        decoder.read_u32_into::<BigEndian>(&mut metadata_buf)?;

        let mut imgs = Vec::new();
        decoder.read_to_end(&mut imgs)?;
        if metadata_buf[0] != IMG_MAGIC_NO {
            Err(Error::new(ErrorKind::InvalidData,
                "Unable to verify MNIST data. Force redownload."))
        } else {
            Ok((metadata_buf, imgs.chunks(784).map(|x| x.into()).collect()))
        }
        
    }

    fn get_decoder<P: AsRef<Path>>(&self, archive: P) -> io::Result<GzDecoder<File>> {
        let archive = File::open(archive)?;
        Ok(GzDecoder::new(archive))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::remove_dir_all;

    #[test]
    #[ignore]
    fn test_builder() {
        let builder = MNISTBuilder::new();
        let mnist = builder.data_home("MNIST").get_data().unwrap();
        assert_eq!(mnist.train_imgs.len(), 60000);
        remove_dir_all("MNIST").unwrap();
    }

}
