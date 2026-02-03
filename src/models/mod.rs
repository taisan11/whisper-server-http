use crate::services::vad::SpeechSegment;
use std::fmt;

#[derive(Debug, Clone)]
pub struct TranscriptionSegment {
    pub start_time: f64,
    pub end_time: f64,
    pub text: String,
}

impl nojson::DisplayJson for TranscriptionSegment {
    fn fmt(&self, f: &mut nojson::JsonFormatter<'_, '_>) -> std::fmt::Result {
        f.object(|f| {
            f.member("start_time", self.start_time)?;
            f.member("end_time", self.end_time)?;
            f.member("text", &self.text)
        })
    }
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub segments: Vec<TranscriptionSegment>,
    pub full_text: String,
    pub vad_segments: Option<Vec<SpeechSegment>>,
}

impl nojson::DisplayJson for TranscriptionResult {
    fn fmt(&self, f: &mut nojson::JsonFormatter<'_, '_>) -> std::fmt::Result {
        f.object(|f| {
            f.member("segments", &self.segments)?;
            f.member("full_text", &self.full_text)?;
            if let Some(ref vad_segs) = self.vad_segments {
                f.member("vad_segments", vad_segs)?;
            }
            Ok(())
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptionStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

impl fmt::Display for TranscriptionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TranscriptionStatus::Pending => write!(f, "pending"),
            TranscriptionStatus::Processing => write!(f, "processing"),
            TranscriptionStatus::Completed => write!(f, "completed"),
            TranscriptionStatus::Failed => write!(f, "failed"),
        }
    }
}

impl nojson::DisplayJson for TranscriptionStatus {
    fn fmt(&self, f: &mut nojson::JsonFormatter<'_, '_>) -> std::fmt::Result {
        f.value(&self.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct TranscriptionJob {
    pub filename: String,
    pub status: TranscriptionStatus,
    pub progress: f32,
    pub error: Option<String>,
}

impl nojson::DisplayJson for TranscriptionJob {
    fn fmt(&self, f: &mut nojson::JsonFormatter<'_, '_>) -> std::fmt::Result {
        f.object(|f| {
            f.member("filename", &self.filename)?;
            f.member("status", &self.status)?;
            f.member("progress", self.progress)?;
            if let Some(ref error) = self.error {
                f.member("error", error)?;
            }
            Ok(())
        })
    }
}

#[derive(Debug)]
pub struct ErrorResponse {
    pub error: String,
}

impl nojson::DisplayJson for ErrorResponse {
    fn fmt(&self, f: &mut nojson::JsonFormatter<'_, '_>) -> std::fmt::Result {
        f.object(|f| f.member("error", &self.error))
    }
}
