/// Not actually used yet

use std::path::{Path, PathBuf};
use std::io;
use std::fs::remove_dir_all;

pub fn get_tmp_dir<P: AsRef<Path>>(dset_name: P) -> PathBuf {
    Path::new("tmp").join(dset_name)
}

pub fn remove_tmp_dir<P: AsRef<Path>>(dset_name: P) -> io::Result<()> {
    remove_dir_all(get_tmp_dir(dset_name))
}
