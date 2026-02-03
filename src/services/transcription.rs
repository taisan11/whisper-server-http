use crate::models::{
    TranscriptionJob, TranscriptionResult, TranscriptionSegment, TranscriptionStatus,
};
use crate::services::vad::VadService;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

pub struct TranscriptionService {
    jobs: Arc<RwLock<HashMap<String, TranscriptionJob>>>,
    results: Arc<RwLock<HashMap<String, TranscriptionResult>>>,
    whisper_ctx: Arc<tokio::sync::Mutex<WhisperContext>>,
    vad_service: Option<Arc<VadService>>,
    data_dir: PathBuf,
}

impl TranscriptionService {
    pub fn new(whisper_ctx: WhisperContext, data_dir: PathBuf) -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            results: Arc::new(RwLock::new(HashMap::new())),
            whisper_ctx: Arc::new(tokio::sync::Mutex::new(whisper_ctx)),
            vad_service: None,
            data_dir,
        }
    }

    pub fn with_vad(mut self, vad_service: VadService) -> Self {
        self.vad_service = Some(Arc::new(vad_service));
        self
    }

    pub async fn create_job(&self, filename: String) -> Result<(), String> {
        let mut jobs = self.jobs.write().await;

        if jobs.contains_key(&filename) {
            return Err("Job already exists".to_string());
        }

        let job = TranscriptionJob {
            filename: filename.clone(),
            status: TranscriptionStatus::Pending,
            progress: 0.0,
            error: None,
        };

        jobs.insert(filename, job);
        Ok(())
    }

    pub async fn get_job_status(&self, filename: &str) -> Option<TranscriptionJob> {
        let jobs = self.jobs.read().await;
        jobs.get(filename).cloned()
    }

    pub async fn get_result(&self, filename: &str) -> Option<TranscriptionResult> {
        let results = self.results.read().await;
        results.get(filename).cloned()
    }

    pub async fn start_transcription(&self, filename: String, audio_data: Vec<f32>) {
        let jobs = Arc::clone(&self.jobs);
        let results = Arc::clone(&self.results);
        let whisper_ctx = Arc::clone(&self.whisper_ctx);
        let vad_service = self.vad_service.clone();
        let data_dir = self.data_dir.clone();

        tokio::spawn(async move {
            info!("Starting transcription for: {}", filename);

            // Update status to processing
            {
                let mut jobs_map = jobs.write().await;
                if let Some(job) = jobs_map.get_mut(&filename) {
                    job.status = TranscriptionStatus::Processing;
                    job.progress = 0.0;
                }
            }

            // Apply VAD if available
            let (processed_audio, vad_segments) = if let Some(vad) = &vad_service {
                info!("Running VAD for: {}", filename);
                match vad.detect_speech_segments(&audio_data, 16000).await {
                    Ok(segments) => {
                        info!(
                            "VAD detected {} speech segments for {}",
                            segments.len(),
                            filename
                        );
                        for (i, seg) in segments.iter().enumerate() {
                            info!(
                                "  VAD segment {}: {:.2}s -> {:.2}s",
                                i + 1,
                                seg.start,
                                seg.end
                            );
                        }
                        let audio = vad.extract_speech_audio(&audio_data, &segments, 16000);
                        info!(
                            "VAD extracted {} samples from {} original samples",
                            audio.len(),
                            audio_data.len()
                        );
                        (audio, Some(segments))
                    }
                    Err(e) => {
                        error!("VAD failed for {}: {}, using original audio", filename, e);
                        (audio_data, None)
                    }
                }
            } else {
                info!("VAD not enabled, using original audio");
                (audio_data, None)
            };

            // Perform transcription
            let result = Self::transcribe(
                whisper_ctx,
                processed_audio,
                filename.clone(),
                Arc::clone(&jobs),
                vad_segments,
            )
            .await;

            match result {
                Ok(transcription_result) => {
                    info!("Transcription completed for: {}", filename);

                    // Save result to JSON file
                    let json_path = data_dir.join(format!("{}.json", filename));
                    if let Err(e) = Self::save_result_to_file(&transcription_result, &json_path) {
                        error!("Failed to save result to file: {}", e);
                    }

                    // Update job status
                    {
                        let mut jobs_map = jobs.write().await;
                        if let Some(job) = jobs_map.get_mut(&filename) {
                            job.status = TranscriptionStatus::Completed;
                            job.progress = 100.0;
                        }
                    }

                    // Store result
                    {
                        let mut results_map = results.write().await;
                        results_map.insert(filename.clone(), transcription_result);
                    }
                }
                Err(e) => {
                    error!("Transcription failed for {}: {}", filename, e);

                    // Update job status
                    let mut jobs_map = jobs.write().await;
                    if let Some(job) = jobs_map.get_mut(&filename) {
                        job.status = TranscriptionStatus::Failed;
                        job.error = Some(e);
                    }
                }
            }
        });
    }

    async fn transcribe(
        whisper_ctx: Arc<tokio::sync::Mutex<WhisperContext>>,
        audio_data: Vec<f32>,
        filename: String,
        jobs: Arc<RwLock<HashMap<String, TranscriptionJob>>>,
        vad_segments: Option<Vec<crate::services::vad::SpeechSegment>>,
    ) -> Result<TranscriptionResult, String> {
        let whisper = whisper_ctx.lock().await;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_language(Some("ja"));

        let mut state = whisper
            .create_state()
            .map_err(|e| format!("Failed to create state: {}", e))?;

        // Run transcription
        state
            .full(params, &audio_data)
            .map_err(|e| format!("Transcription failed: {}", e))?;

        let num_segments = state.full_n_segments();
        let mut segments = Vec::new();
        let mut full_text = String::new();

        for i in 0..num_segments {
            // Update progress
            let progress = (i as f32 / num_segments as f32) * 100.0;
            {
                let mut jobs_map = jobs.write().await;
                if let Some(job) = jobs_map.get_mut(&filename) {
                    job.progress = progress;
                }
            }

            if let Some(segment) = state.get_segment(i) {
                let segment_text = segment.to_str().unwrap_or("");
                let start_time = segment.start_timestamp() as f64 / 100.0;
                let end_time = segment.end_timestamp() as f64 / 100.0;

                info!(
                    "[{}] [{:.2}s -> {:.2}s] {}",
                    filename, start_time, end_time, segment_text
                );

                segments.push(TranscriptionSegment {
                    start_time,
                    end_time,
                    text: segment_text.trim().to_string(),
                });

                if !full_text.is_empty() {
                    full_text.push(' ');
                }
                full_text.push_str(segment_text.trim());
            }
        }

        Ok(TranscriptionResult {
            segments,
            full_text,
            vad_segments,
        })
    }

    fn save_result_to_file(result: &TranscriptionResult, path: &PathBuf) -> Result<(), String> {
        let json = nojson::json(|f| {
            f.set_spacing(true);
            f.set_indent_size(2);
            f.value(result)
        });

        std::fs::write(path, json.to_string())
            .map_err(|e| format!("Failed to write file: {}", e))?;

        Ok(())
    }

    pub async fn remove_job(&self, filename: &str) -> bool {
        let mut jobs = self.jobs.write().await;
        let mut results = self.results.write().await;

        jobs.remove(filename);
        results.remove(filename).is_some()
    }
}
