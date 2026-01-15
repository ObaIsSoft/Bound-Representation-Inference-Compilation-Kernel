use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use std::fs;
use chrono::{DateTime, Utc};
use thiserror::Error;
use walkdir::WalkDir;

#[derive(Error, Debug, Serialize)]
pub enum FsError {
    #[error("IO Error: {0}")]
    Io(String),
    #[error("Not Found: {0}")]
    NotFound(String),
    #[error("Permission Denied: {0}")]
    PermissionDenied(String),
}

impl From<std::io::Error> for FsError {
    fn from(err: std::io::Error) -> Self {
        FsError::Io(err.to_string())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FileEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub size: u64,
    pub created: Option<String>,
    pub modified: Option<String>,
    pub children: Option<Vec<FileEntry>>, // For recursive structure if needed
}

pub struct FsCore;

impl FsCore {
    /// Read file content as string
    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<String, FsError> {
        fs::read_to_string(path).map_err(FsError::from)
    }

    /// Write content to file (overwrites)
    pub fn write_file<P: AsRef<Path>>(path: P, content: String) -> Result<(), FsError> {
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, content).map_err(FsError::from)
    }

    /// List directory contents (non-recursive)
    pub fn list_dir<P: AsRef<Path>>(path: P) -> Result<Vec<FileEntry>, FsError> {
        let entries = fs::read_dir(&path).map_err(FsError::from)?;
        let mut results = Vec::new();

        for entry in entries {
            let entry = entry.map_err(FsError::from)?;
            let metadata = entry.metadata().map_err(FsError::from)?;
            
            let created = metadata.created().ok()
                .map(|t| DateTime::<Utc>::from(t).to_rfc3339());
            let modified = metadata.modified().ok()
                .map(|t| DateTime::<Utc>::from(t).to_rfc3339());

            results.push(FileEntry {
                name: entry.file_name().to_string_lossy().to_string(),
                path: entry.path().to_string_lossy().to_string(),
                is_dir: metadata.is_dir(),
                size: metadata.len(),
                created,
                modified,
                children: None,
            });
        }
        
        // Sort: directories first, then alphabetical
        results.sort_by(|a, b| {
            if a.is_dir == b.is_dir {
                a.name.cmp(&b.name)
            } else {
                b.is_dir.cmp(&a.is_dir)
            }
        });

        Ok(results)
    }

    /// Delete file or directory (recursively if dir)
    pub fn delete<P: AsRef<Path>>(path: P) -> Result<(), FsError> {
        let path = path.as_ref();
        if path.is_dir() {
            fs::remove_dir_all(path).map_err(FsError::from)
        } else {
            fs::remove_file(path).map_err(FsError::from)
        }
    }
    
    /// Create directory
    pub fn create_dir<P: AsRef<Path>>(path: P) -> Result<(), FsError> {
        fs::create_dir_all(path).map_err(FsError::from)
    }
}
