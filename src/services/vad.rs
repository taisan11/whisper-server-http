use ort::{
    session::{Session, SessionInputValue},
    tensor::TensorElementType,
    value::{Value, ValueType},
};
use std::io::{Error as IoError, ErrorKind};
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
struct StateInputSpec {
    name: String,
    shape: Vec<i64>,
    len: usize,
}

#[derive(Debug, Clone)]
struct VadInputLayout {
    audio_input: String,
    audio_input_samples: Option<usize>,
    state_inputs: Vec<StateInputSpec>,
    sample_rate_input: Option<String>,
}

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
    input_layout: VadInputLayout,
}

impl VadService {
    pub fn new(model_path: &Path, threshold: f32) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Loading Silero VAD model from: {}", model_path.display());

        let session = Session::builder()?.commit_from_file(model_path)?;
        let input_layout = Self::infer_input_layout(&session)?;

        info!("Silero VAD model loaded successfully!");

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            threshold,
            input_layout,
        })
    }

    fn normalize_shape(shape: &[i64]) -> Vec<i64> {
        if shape.is_empty() {
            return vec![1];
        }

        shape
            .iter()
            .map(|&dim| if dim > 0 { dim } else { 1 })
            .collect()
    }

    fn infer_input_layout(session: &Session) -> Result<VadInputLayout, Box<dyn std::error::Error>> {
        let mut float_inputs: Vec<(String, Vec<i64>, usize, Option<usize>)> = Vec::new();
        let mut int64_inputs: Vec<String> = Vec::new();

        for input in session.inputs() {
            info!("VAD input '{}': {}", input.name(), input.dtype());

            if let ValueType::Tensor { ty, shape, .. } = input.dtype() {
                if *ty == TensorElementType::Float32 {
                    let raw_shape = shape.as_ref();
                    let inferred_samples = if raw_shape.len() >= 2 && raw_shape[1] > 0 {
                        Some(raw_shape[1] as usize)
                    } else if raw_shape.len() == 1 && raw_shape[0] > 0 {
                        Some(raw_shape[0] as usize)
                    } else {
                        None
                    };

                    float_inputs.push((
                        input.name().to_string(),
                        Self::normalize_shape(raw_shape),
                        raw_shape.len(),
                        inferred_samples,
                    ));
                } else if *ty == TensorElementType::Int64 {
                    int64_inputs.push(input.name().to_string());
                }
            }
        }

        if float_inputs.is_empty() {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                "VAD model has no float32 input tensor",
            )
            .into());
        }

        // Audio input is typically the only rank<=2 float tensor.
        let audio_idx = float_inputs
            .iter()
            .position(|(_, _, rank, _)| *rank <= 2)
            .unwrap_or(0);
        let audio_input = float_inputs[audio_idx].0.clone();
        let audio_input_samples = float_inputs[audio_idx].3;

        let mut state_inputs = Vec::new();
        for (idx, (name, shape, _, _)) in float_inputs.into_iter().enumerate() {
            if idx == audio_idx {
                continue;
            }

            let len = shape
                .iter()
                .fold(1usize, |acc, &dim| acc.saturating_mul(dim as usize))
                .max(1);

            state_inputs.push(StateInputSpec { name, shape, len });
        }

        let sample_rate_input = int64_inputs.into_iter().next();

        info!(
            "VAD input layout: audio='{}', audio_samples={:?}, state_tensors={}, sample_rate={}",
            audio_input,
            audio_input_samples,
            state_inputs.len(),
            sample_rate_input.as_deref().unwrap_or("none")
        );
        for state in &state_inputs {
            debug!("VAD state input '{}' shape={:?}", state.name, state.shape);
        }

        Ok(VadInputLayout {
            audio_input,
            audio_input_samples,
            state_inputs,
            sample_rate_input,
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
        let (speech_probs, window_size_samples) = self.process_audio(&audio_16k).await?;

        // Convert probabilities to speech segments
        let segments =
            self.probabilities_to_segments(&speech_probs, SAMPLE_RATE, window_size_samples);

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

    async fn process_audio(
        &self,
        audio: &[f32],
    ) -> Result<(Vec<f32>, usize), Box<dyn std::error::Error>> {
        // Silero v6 uses a 64-sample left context with 512-sample windows (input length 576).
        // Older variants use 512 directly.
        let (window_size_samples, context_size_samples) =
            match self.input_layout.audio_input_samples {
                Some(model_input_samples) if model_input_samples > WINDOW_SIZE_SAMPLES => (
                    WINDOW_SIZE_SAMPLES,
                    model_input_samples - WINDOW_SIZE_SAMPLES,
                ),
                Some(model_input_samples) if model_input_samples > 0 => (model_input_samples, 0),
                _ => (WINDOW_SIZE_SAMPLES, 0),
            };

        let num_samples = audio.len();
        let num_windows = (num_samples + window_size_samples - 1) / window_size_samples;

        let mut speech_probs = Vec::with_capacity(num_windows);
        let mut context = vec![0.0f32; context_size_samples];

        // Initialize recurrent state tensors based on model signature.
        let mut state_tensors: Vec<Vec<f32>> = self
            .input_layout
            .state_inputs
            .iter()
            .map(|state| vec![0.0f32; state.len])
            .collect();

        for window_idx in 0..num_windows {
            let start = window_idx * window_size_samples;
            let end = (start + window_size_samples).min(num_samples);
            let window_len = end - start;

            // Prepare input window, pad if necessary
            let mut window = vec![0.0f32; window_size_samples];
            window[..window_len].copy_from_slice(&audio[start..end]);

            let mut model_input_window =
                Vec::with_capacity(context_size_samples.saturating_add(window_size_samples));
            if context_size_samples > 0 {
                model_input_window.extend_from_slice(&context);
            }
            model_input_window.extend_from_slice(&window);

            // Create input tensor (1, N)
            let input_value =
                Value::from_array(([1, model_input_window.len()], model_input_window.clone()))?;

            let mut model_inputs: Vec<(String, SessionInputValue<'_>)> = Vec::with_capacity(
                1 + self.input_layout.state_inputs.len()
                    + usize::from(self.input_layout.sample_rate_input.is_some()),
            );
            model_inputs.push((self.input_layout.audio_input.clone(), input_value.into()));

            for (state_spec, state_data) in self
                .input_layout
                .state_inputs
                .iter()
                .zip(state_tensors.iter())
            {
                let state_value =
                    Value::from_array((state_spec.shape.clone(), state_data.clone()))?;
                model_inputs.push((state_spec.name.clone(), state_value.into()));
            }

            if let Some(sample_rate_input) = &self.input_layout.sample_rate_input {
                let sr_value = Value::from_array(([1], vec![SAMPLE_RATE as i64]))?;
                model_inputs.push((sample_rate_input.clone(), sr_value.into()));
            }

            // Run inference and extract outputs while session lock is held
            let (prob, next_states) = {
                let mut session = self.session.lock().await;
                let outputs = session.run(model_inputs)?;

                // Extract output probability
                let (_output_shape, output_data) = outputs[0].try_extract_tensor::<f32>()?;
                let prob = *output_data.first().unwrap_or(&0.0);

                let mut extracted_states = Vec::with_capacity(self.input_layout.state_inputs.len());
                for (state_idx, state_spec) in self.input_layout.state_inputs.iter().enumerate() {
                    let output_idx = state_idx + 1;
                    if output_idx >= outputs.len() {
                        warn!(
                            "VAD state output #{} missing for '{}'; keeping previous state",
                            output_idx, state_spec.name
                        );
                        extracted_states.push(None);
                        continue;
                    }

                    let (_shape, state_data) = outputs[output_idx].try_extract_tensor::<f32>()?;
                    let mut state_vec = state_data.to_vec();

                    if state_vec.len() != state_spec.len {
                        warn!(
                            "VAD state size mismatch for '{}': expected {}, got {}",
                            state_spec.name,
                            state_spec.len,
                            state_vec.len()
                        );
                        state_vec.truncate(state_spec.len);
                        state_vec.resize(state_spec.len, 0.0);
                    }

                    extracted_states.push(Some(state_vec));
                }

                (prob, extracted_states)
            };

            speech_probs.push(prob);

            // Update recurrent state tensors from extracted outputs.
            for (state_idx, maybe_state) in next_states.into_iter().enumerate() {
                if let Some(state) = maybe_state {
                    if state_idx < state_tensors.len() {
                        state_tensors[state_idx] = state;
                    }
                }
            }

            if context_size_samples > 0 && context_size_samples <= model_input_window.len() {
                let tail_start = model_input_window.len() - context_size_samples;
                context.copy_from_slice(&model_input_window[tail_start..]);
            }

            if window_idx % 100 == 0 {
                debug!("Processed {}/{} windows", window_idx + 1, num_windows);
            }
        }

        Ok((speech_probs, window_size_samples))
    }

    fn probabilities_to_segments(
        &self,
        probs: &[f32],
        sample_rate: usize,
        window_size_samples: usize,
    ) -> Vec<SpeechSegment> {
        let mut segments = Vec::new();
        let mut in_speech = false;

        let window_duration_ms = ((window_size_samples * 1000) / sample_rate).max(1);
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
                            (temp_speech_start * window_size_samples) as f64 / sample_rate as f64;
                        let end_time = ((i - silence_counter + pad_windows).min(probs.len())
                            * window_size_samples) as f64
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
                    (temp_speech_start * window_size_samples) as f64 / sample_rate as f64;
                let end_time = (probs.len() * window_size_samples) as f64 / sample_rate as f64;

                segments.push(SpeechSegment {
                    start: start_time,
                    end: end_time,
                });
            }
        }

        segments
    }

    fn resample_audio(&self, audio: &[f32], from_rate: usize, to_rate: usize) -> Vec<f32> {
        if audio.is_empty() || from_rate == to_rate {
            return audio.to_vec();
        }
        if from_rate == 0 || to_rate == 0 {
            return Vec::new();
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
}
