pub mod finish;
pub mod status;
pub mod upload;

pub use finish::finish_handler;
pub use status::status_handler;
pub use upload::upload_handler;

const MAX_FILENAME_LEN: usize = 128;

pub(crate) fn validate_filename(filename: &str) -> Result<String, String> {
    let trimmed = filename.trim();
    if trimmed.is_empty() {
        return Err("filename must not be empty".to_string());
    }

    if trimmed.len() > MAX_FILENAME_LEN {
        return Err(format!(
            "filename is too long (max {} characters)",
            MAX_FILENAME_LEN
        ));
    }

    if trimmed == "." || trimmed == ".." {
        return Err("filename is invalid".to_string());
    }

    if !trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
    {
        return Err("filename contains invalid characters".to_string());
    }

    let path = std::path::Path::new(trimmed);
    if path.is_absolute() || path.components().count() != 1 {
        return Err("filename must not contain path separators".to_string());
    }

    Ok(trimmed.to_string())
}
