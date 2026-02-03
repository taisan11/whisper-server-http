use ndarray::Array2;
use ort::{session::Session, value::Value};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

const SAMPLE_RATE: usize = 16000;
const WINDOW_SIZE_SAMPLES: usize = 512; // 32ms at 16kHz
const MIN_SPEECH_DURATION_MS: usize = 250;
const MIN_SILENCE_DURATION_MS: usize = 100;
const SPEECH_PAD_MS: usize = 30;

#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub start: f64,
    pub end: f64,
}

impl nojson::DisplayJson for SpeechSegment {
    fn fmt(&self, f: &mut nojson::JsonFormatter<'_, '_>) -> std::fmt::Result {
        f.object(|f| {
            f.member("start", self.start)?;
            f.member("end", self.end)
        })
    }
}

pub struct VadService {
    session: Arc<Mutex<Session>>,
    threshold: f32,
}

impl VadService {
    pub fn new(model_path: &Path, threshold: f32) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Loading Silero VAD model from: {}", model_path.display());

        let session = Session::builder()?.commit_from_file(model_path)?;

        info!("Silero VAD model loaded successfully!");

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            threshold,
        })
    }

    /// Process audio and detect speech segments
    pub async fn detect_speech_segments(
        &self,
        audio_data: &[f32],
        sample_rate: usize,
    ) -> Result<Vec<SpeechSegment>, Box<dyn std::error::Error>> {
        info!(
            "Starting VAD processing for {} samples at {}Hz",
            audio_data.len(),
            sample_rate
        );

        // Resample to 16kHz if needed
        let audio_16k = if sample_rate != SAMPLE_RATE {
            info!("Resampling from {}Hz to {}Hz", sample_rate, SAMPLE_RATE);
            self.resample_audio(audio_data, sample_rate, SAMPLE_RATE)
        } else {
            audio_data.to_vec()
        };

        // Run VAD on the audio
        let speech_probs = self.process_audio(&audio_16k).await?;

        // Convert probabilities to speech segments
        let segments = self.probabilities_to_segments(&speech_probs, SAMPLE_RATE);

        info!("Detected {} speech segments", segments.len());
        for (i, segment) in segments.iter().enumerate() {
            debug!(
                "Segment {}: {:.2}s -> {:.2}s",
                i + 1,
                segment.start,
                segment.end
            );
        }

        Ok(segments)
    }

    async fn process_audio(&self, audio: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let num_samples = audio.len();
        let num_windows = (num_samples + WINDOW_SIZE_SAMPLES - 1) / WINDOW_SIZE_SAMPLES;

        let mut speech_probs = Vec::with_capacity(num_windows);

        // Initialize state tensors (2, 1, 128) for h and c
        let mut h = Array2::<f32>::zeros((2, 128));
        let mut c = Array2::<f32>::zeros((2, 128));

        for window_idx in 0..num_windows {
            let start = window_idx * WINDOW_SIZE_SAMPLES;
            let end = (start + WINDOW_SIZE_SAMPLES).min(num_samples);
            let window_len = end - start;

            // Prepare input window, pad if necessary
            let mut window = vec![0.0f32; WINDOW_SIZE_SAMPLES];
            window[..window_len].copy_from_slice(&audio[start..end]);

            // Create input tensor (1, 512) using shape+vec tuple
            let input_value = Value::from_array(([1, WINDOW_SIZE_SAMPLES], window))?;

            // Reshape state tensors to (2, 1, 128) using shape+vec tuple
            let h_data: Vec<f32> = h.iter().copied().collect();
            let h_value = Value::from_array(([2, 1, 128], h_data))?;

            let c_data: Vec<f32> = c.iter().copied().collect();
            let c_value = Value::from_array(([2, 1, 128], c_data))?;

            let sr_value = Value::from_array(([1], vec![SAMPLE_RATE as i64]))?;

            // Run inference and extract outputs while session lock is held
            let (prob, h_data, c_data) = {
                let mut session = self.session.lock().await;
                let outputs = session.run(ort::inputs![input_value, h_value, c_value, sr_value])?;

                // Extract output probability
                let (_output_shape, output_data) = outputs[0].try_extract_tensor::<f32>()?;
                let prob = *output_data.first().unwrap_or(&0.0);

                // Extract state tensors
                let (h_shape, h_data) = outputs[1].try_extract_tensor::<f32>()?;
                let h_vec = if h_shape.as_ref().len() == 3
                    && h_shape[0] == 2
                    && h_shape[1] == 1
                    && h_shape[2] == 128
                {
                    h_data.to_vec()
                } else {
                    vec![]
                };

                let (c_shape, c_data) = outputs[2].try_extract_tensor::<f32>()?;
                let c_vec = if c_shape.as_ref().len() == 3
                    && c_shape[0] == 2
                    && c_shape[1] == 1
                    && c_shape[2] == 128
                {
                    c_data.to_vec()
                } else {
                    vec![]
                };

                (prob, h_vec, c_vec)
            };

            speech_probs.push(prob);

            // Update state tensors from extracted data
            if !h_data.is_empty() {
                for i in 0..2 {
                    for j in 0..128 {
                        let idx = i * 128 + j;
                        if idx < h_data.len() {
                            h[[i, j]] = h_data[idx];
                        }
                    }
                }
            }

            if !c_data.is_empty() {
                for i in 0..2 {
                    for j in 0..128 {
                        let idx = i * 128 + j;
                        if idx < c_data.len() {
                            c[[i, j]] = c_data[idx];
                        }
                    }
                }
            }

            if window_idx % 100 == 0 {
                debug!("Processed {}/{} windows", window_idx + 1, num_windows);
            }
        }

        Ok(speech_probs)
    }

    fn probabilities_to_segments(&self, probs: &[f32], sample_rate: usize) -> Vec<SpeechSegment> {
        let mut segments = Vec::new();
        let mut in_speech = false;

        let window_duration_ms = (WINDOW_SIZE_SAMPLES * 1000) / sample_rate;
        let min_speech_windows = MIN_SPEECH_DURATION_MS / window_duration_ms;
        let min_silence_windows = MIN_SILENCE_DURATION_MS / window_duration_ms;
        let pad_windows = SPEECH_PAD_MS / window_duration_ms;

        let mut silence_counter = 0;
        let mut temp_speech_start = 0;

        for (i, &prob) in probs.iter().enumerate() {
            let is_speech = prob >= self.threshold;

            if is_speech {
                if !in_speech {
                    // Start of potential speech segment
                    temp_speech_start = i.saturating_sub(pad_windows);
                    in_speech = true;
                    silence_counter = 0;
                } else {
                    // Continue speech, reset silence counter
                    silence_counter = 0;
                }
            } else if in_speech {
                // In speech but current window is silence
                silence_counter += 1;

                if silence_counter >= min_silence_windows {
                    // End of speech segment
                    let speech_length = i - temp_speech_start - silence_counter;

                    if speech_length >= min_speech_windows {
                        let start_time =
                            (temp_speech_start * WINDOW_SIZE_SAMPLES) as f64 / sample_rate as f64;
                        let end_time = ((i - silence_counter + pad_windows).min(probs.len())
                            * WINDOW_SIZE_SAMPLES) as f64
                            / sample_rate as f64;

                        segments.push(SpeechSegment {
                            start: start_time,
                            end: end_time,
                        });
                    }

                    in_speech = false;
                    silence_counter = 0;
                }
            }
        }

        // Handle ongoing speech at the end
        if in_speech {
            let speech_length = probs.len() - temp_speech_start - silence_counter;
            if speech_length >= min_speech_windows {
                let start_time =
                    (temp_speech_start * WINDOW_SIZE_SAMPLES) as f64 / sample_rate as f64;
                let end_time = (probs.len() * WINDOW_SIZE_SAMPLES) as f64 / sample_rate as f64;

                segments.push(SpeechSegment {
                    start: start_time,
                    end: end_time,
                });
            }
        }

        segments
    }

    fn resample_audio(&self, audio: &[f32], from_rate: usize, to_rate: usize) -> Vec<f32> {
        if from_rate == to_rate {
            return audio.to_vec();
        }

        let ratio = to_rate as f64 / from_rate as f64;
        let output_len = (audio.len() as f64 * ratio) as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f64 / ratio;
            let src_idx_floor = src_idx.floor() as usize;
            let src_idx_ceil = (src_idx_floor + 1).min(audio.len() - 1);
            let frac = src_idx - src_idx_floor as f64;

            let sample = if src_idx_floor >= audio.len() {
                0.0
            } else {
                let a = audio[src_idx_floor];
                let b = audio[src_idx_ceil];
                a + (b - a) * frac as f32
            };

            output.push(sample);
        }

        output
    }

    /// Extract audio segments based on detected speech
    pub fn extract_speech_audio(
        &self,
        audio_data: &[f32],
        segments: &[SpeechSegment],
        sample_rate: usize,
    ) -> Vec<f32> {
        if segments.is_empty() {
            warn!("No speech segments detected, returning original audio");
            return audio_data.to_vec();
        }

        let mut result = Vec::new();

        for segment in segments {
            let start_sample = (segment.start * sample_rate as f64) as usize;
            let end_sample = (segment.end * sample_rate as f64) as usize;

            let start_sample = start_sample.min(audio_data.len());
            let end_sample = end_sample.min(audio_data.len());

            if start_sample < end_sample {
                result.extend_from_slice(&audio_data[start_sample..end_sample]);
            }
        }

        info!(
            "Extracted {} samples from {} segments",
            result.len(),
            segments.len()
        );
        result
    }
}
