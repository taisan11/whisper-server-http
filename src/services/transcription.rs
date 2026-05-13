use crate::models::{
    TranscriptionJob, TranscriptionResult, TranscriptionSegment, TranscriptionStatus,
};
use crate::services::vad::{SpeechSegment, VadService};
use std::collections::{HashMap, VecDeque};
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

const MAX_FINISHED_JOBS: usize = 200;
const SAMPLE_RATE: usize = 16000;
const BLOCK_SECONDS: usize = 30;
const BLOCK_SAMPLES: usize = SAMPLE_RATE * BLOCK_SECONDS;
const BLOCK_EDGE_TRIM_SECONDS: f64 = 0.5;
const NG_WORDS: [&str; 3] = ["あ", "ん", "ご視聴ありがとうございました"];

#[derive(Debug, Clone, Copy)]
struct AudioRange {
    start_sample: usize,
    end_sample: usize,
}

pub struct TranscriptionService {
    jobs: Arc<RwLock<HashMap<String, TranscriptionJob>>>,
    results: Arc<RwLock<HashMap<String, TranscriptionResult>>>,
    finished_jobs: Arc<RwLock<VecDeque<String>>>,
    whisper_ctx: Arc<tokio::sync::Mutex<WhisperContext>>,
    vad_service: Option<Arc<VadService>>,
    data_dir: PathBuf,
}

impl TranscriptionService {
    pub fn new(whisper_ctx: WhisperContext, data_dir: PathBuf) -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            results: Arc::new(RwLock::new(HashMap::new())),
            finished_jobs: Arc::new(RwLock::new(VecDeque::new())),
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
        let finished_jobs = Arc::clone(&self.finished_jobs);
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
            let vad_segments = if let Some(vad) = &vad_service {
                info!("Running VAD for: {}", filename);
                match vad.detect_speech_segments(&audio_data, SAMPLE_RATE).await {
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

                        if !segments.is_empty() {
                            let speech_samples = segments
                                .iter()
                                .map(|seg| {
                                    let start = (seg.start * SAMPLE_RATE as f64) as usize;
                                    let end = (seg.end * SAMPLE_RATE as f64) as usize;
                                    end.saturating_sub(start)
                                })
                                .sum::<usize>();
                            info!(
                                "VAD selected ~{} speech samples from {} original samples",
                                speech_samples,
                                audio_data.len()
                            );
                        }

                        Some(segments)
                    }
                    Err(e) => {
                        error!("VAD failed for {}: {}, using original audio", filename, e);
                        None
                    }
                }
            } else {
                info!("VAD not enabled, using original audio");
                None
            };

            // Perform transcription
            let result = Self::transcribe(
                whisper_ctx,
                audio_data,
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

                    Self::record_finished_job(
                        filename.clone(),
                        Arc::clone(&finished_jobs),
                        Arc::clone(&jobs),
                        Arc::clone(&results),
                        data_dir.clone(),
                    )
                    .await;
                }
                Err(e) => {
                    error!("Transcription failed for {}: {}", filename, e);

                    // Update job status
                    let mut jobs_map = jobs.write().await;
                    if let Some(job) = jobs_map.get_mut(&filename) {
                        job.status = TranscriptionStatus::Failed;
                        job.error = Some(e);
                    }

                    Self::record_finished_job(
                        filename.clone(),
                        Arc::clone(&finished_jobs),
                        Arc::clone(&jobs),
                        Arc::clone(&results),
                        data_dir.clone(),
                    )
                    .await;
                }
            }
        });
    }

    async fn transcribe(
        whisper_ctx: Arc<tokio::sync::Mutex<WhisperContext>>,
        audio_data: Vec<f32>,
        filename: String,
        jobs: Arc<RwLock<HashMap<String, TranscriptionJob>>>,
        vad_segments: Option<Vec<SpeechSegment>>,
    ) -> Result<TranscriptionResult, String> {
        let whisper = whisper_ctx.lock().await;
        let mut state = whisper
            .create_state()
            .map_err(|e| format!("Failed to create state: {}", e))?;
        let ranges = Self::build_audio_ranges(audio_data.len(), vad_segments.as_deref());
        let total_target_samples = ranges
            .iter()
            .map(|range| range.end_sample.saturating_sub(range.start_sample))
            .sum::<usize>()
            .max(1);

        let mut segments = Vec::new();
        let mut full_text = String::new();
        let mut completed_samples = 0usize;

        for range in ranges {
            let range_total = range.end_sample.saturating_sub(range.start_sample);
            if range_total == 0 {
                continue;
            }

            let mut cursor = range.start_sample;
            while cursor < range.end_sample {
                let block_end = (cursor + BLOCK_SAMPLES).min(range.end_sample);
                let block = &audio_data[cursor..block_end];
                let block_offset_sec = cursor as f64 / SAMPLE_RATE as f64;
                let block_duration_sec = (block_end - cursor) as f64 / SAMPLE_RATE as f64;

                state
                    .full(Self::create_whisper_params(), block)
                    .map_err(|e| format!("Transcription failed: {}", e))?;

                let num_segments = state.full_n_segments();
                let mut block_segments = Vec::new();
                for i in 0..num_segments {
                    if let Some(segment) = state.get_segment(i) {
                        let text = segment.to_str().unwrap_or("").trim().to_string();
                        if text.is_empty() || Self::is_ng_word(&text) {
                            continue;
                        }

                        let start_time =
                            block_offset_sec + segment.start_timestamp() as f64 / 100.0;
                        let end_time = block_offset_sec + segment.end_timestamp() as f64 / 100.0;
                        if end_time <= start_time {
                            continue;
                        }

                        block_segments.push(TranscriptionSegment {
                            start_time,
                            end_time,
                            text,
                        });
                    }
                }

                if block_segments.len() > 1 {
                    if let Some(last) = block_segments.last() {
                        let last_end_local = (last.end_time - block_offset_sec).max(0.0);
                        if block_duration_sec - last_end_local < BLOCK_EDGE_TRIM_SECONDS {
                            block_segments.pop();
                        }
                    }
                }

                let mut block_max_end_abs: Option<f64> = None;
                for segment in block_segments {
                    info!(
                        "[{}] [{:.2}s -> {:.2}s] {}",
                        filename, segment.start_time, segment.end_time, segment.text
                    );

                    if !full_text.is_empty() {
                        full_text.push(' ');
                    }
                    full_text.push_str(&segment.text);
                    block_max_end_abs = Some(
                        block_max_end_abs
                            .map_or(segment.end_time, |current| current.max(segment.end_time)),
                    );
                    segments.push(segment);
                }

                let mut next_cursor = block_end;
                if let Some(max_end) = block_max_end_abs {
                    let candidate = (max_end * SAMPLE_RATE as f64) as usize;
                    if candidate > cursor && candidate < block_end {
                        next_cursor = candidate;
                    }
                }
                if next_cursor <= cursor {
                    next_cursor = block_end;
                }

                let range_processed = next_cursor
                    .saturating_sub(range.start_sample)
                    .min(range_total);
                let progress = ((completed_samples + range_processed) as f32
                    / total_target_samples as f32)
                    * 100.0;
                Self::update_job_progress(&jobs, &filename, progress.min(99.0)).await;

                cursor = next_cursor;
            }

            completed_samples = completed_samples.saturating_add(range_total);
        }

        Self::update_job_progress(&jobs, &filename, 99.0).await;

        Ok(TranscriptionResult {
            segments,
            full_text,
            vad_segments,
        })
    }

    fn build_audio_ranges(
        total_samples: usize,
        vad_segments: Option<&[SpeechSegment]>,
    ) -> Vec<AudioRange> {
        if total_samples == 0 {
            return Vec::new();
        }

        let mut ranges = Vec::new();
        if let Some(vad_segments) = vad_segments {
            for segment in vad_segments {
                let start = ((segment.start * SAMPLE_RATE as f64) as usize).min(total_samples);
                let end = ((segment.end * SAMPLE_RATE as f64) as usize).min(total_samples);
                if start < end {
                    ranges.push(AudioRange {
                        start_sample: start,
                        end_sample: end,
                    });
                }
            }
        }

        if ranges.is_empty() {
            ranges.push(AudioRange {
                start_sample: 0,
                end_sample: total_samples,
            });
        } else {
            ranges.sort_by_key(|range| range.start_sample);
            let mut merged: Vec<AudioRange> = Vec::with_capacity(ranges.len());
            for range in ranges {
                if let Some(last) = merged.last_mut() {
                    if range.start_sample <= last.end_sample {
                        last.end_sample = last.end_sample.max(range.end_sample);
                    } else {
                        merged.push(range);
                    }
                } else {
                    merged.push(range);
                }
            }
            ranges = merged;
        }

        ranges
    }

    fn create_whisper_params() -> FullParams<'static, 'static> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_no_context(true);
        params.set_language(Some("ja"));

        if let Ok(parallelism) = std::thread::available_parallelism() {
            if let Ok(threads) = i32::try_from(parallelism.get()) {
                params.set_n_threads(threads);
            }
        }

        params
    }

    fn is_ng_word(text: &str) -> bool {
        NG_WORDS.contains(&text)
    }

    async fn update_job_progress(
        jobs: &Arc<RwLock<HashMap<String, TranscriptionJob>>>,
        filename: &str,
        progress: f32,
    ) {
        let mut jobs_map = jobs.write().await;
        if let Some(job) = jobs_map.get_mut(filename) {
            job.progress = progress;
        }
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

    async fn record_finished_job(
        filename: String,
        finished_jobs: Arc<RwLock<VecDeque<String>>>,
        jobs: Arc<RwLock<HashMap<String, TranscriptionJob>>>,
        results: Arc<RwLock<HashMap<String, TranscriptionResult>>>,
        data_dir: PathBuf,
    ) {
        let mut evicted_jobs = Vec::new();

        {
            let mut finished = finished_jobs.write().await;
            finished.push_back(filename);
            while finished.len() > MAX_FINISHED_JOBS {
                if let Some(old_filename) = finished.pop_front() {
                    evicted_jobs.push(old_filename);
                }
            }
        }

        for old_filename in evicted_jobs {
            {
                let mut jobs_map = jobs.write().await;
                jobs_map.remove(&old_filename);
            }
            {
                let mut results_map = results.write().await;
                results_map.remove(&old_filename);
            }
            Self::cleanup_files(&data_dir, &old_filename);
            info!("Evicted old transcription job: {}", old_filename);
        }
    }

    fn cleanup_files(data_dir: &PathBuf, filename: &str) {
        let wav_path = data_dir.join(format!("{}.wav", filename));
        let json_path = data_dir.join(format!("{}.json", filename));

        for path in [wav_path, json_path] {
            if let Err(e) = std::fs::remove_file(&path) {
                if e.kind() != ErrorKind::NotFound {
                    warn!("Failed to remove {}: {}", path.display(), e);
                }
            }
        }
    }
}
