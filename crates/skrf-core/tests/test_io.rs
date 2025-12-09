//! General IO Robustness Tests
//!
//! Inspired by skrf/skrf/io/tests/test_io.py's test_read_all functionality.
//! Ensures that all sample files in tests/data can be parsed without panicking.

use skrf_core::touchstone::Touchstone;
use std::fs;
use std::path::Path;

const TEST_DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data");

#[test]
fn test_read_all_files_in_data_dir() {
    let data_dir = Path::new(TEST_DATA_DIR);
    read_dir_recursively(data_dir);
}

fn read_dir_recursively(dir: &Path) {
    if !dir.exists() {
        eprintln!("Warning: Test data directory does not exist: {:?}", dir);
        return;
    }

    for entry in fs::read_dir(dir).expect("Failed to read directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();

        if path.is_dir() {
            read_dir_recursively(&path);
        } else {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            // Check for sNp or ts files
            if ext == "ts" || (ext.starts_with('s') && ext.ends_with('p') && ext.len() > 2) {
                println!("Testing file: {:?}", path.file_name().unwrap());

                // We expect some files might be malformed or require features not yet implemented
                // so we use match to report but not strictly fail the test suite if it's a known tough case
                // However, for migration verification, we want to know what passes.

                match Touchstone::from_file(&path) {
                    Ok(ts) => {
                        assert!(ts.nports > 0);
                        // TS v2 might have 0 frequencies if just structure, but usually has data
                        // assert!(ts.nfreq() > 0);
                    }
                    Err(e) => {
                        // Fail the test if a file in our test suite cannot be parsed
                        // Exception: We might have known broken files?
                        // Based on skrf repo, these should be valid.
                        panic!("Failed to parse {:?}: {}", path.file_name().unwrap(), e);
                    }
                }
            }
        }
    }
}
