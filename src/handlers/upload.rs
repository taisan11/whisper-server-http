use super::validate_filename;
use crate::models::ErrorResponse;
use crate::services::TranscriptionService;
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use std::sync::{Arc, OnceLock};
use tracing::{error, info, warn};

const DEFAULT_MAX_AUDIO_SAMPLES: usize = 16000 * 60 * 30; // 30 minutes at 16kHz
const MAX_AUDIO_SAMPLES_ENV: &str = "MAX_AUDIO_SAMPLES";
const MIN_SAMPLE_RATE: u32 = 8000;
const MAX_SAMPLE_RATE: u32 = 192000;

pub async fn upload_handler(
    State(service): State<Arc<TranscriptionService>>,
    mut multipart: Multipart,
) -> Response {
    info!("Received upload request");

    let mut audio_data: Option<Vec<f32>> = None;
    let mut sample_rate: u32 = 16000;
    let mut filename: Option<String> = None;

    // Parse multipart form data
    while let Ok(Some(mut field)) = multipart.next_field().await {
        let name = field.name().map(|s| s.to_string()).unwrap_or_default();

        match name.as_str() {
            "audio" => {
                // Get original filename if available
                if let Some(fname) = field.file_name() {
                    filename = Some(fname.to_string());
                }

                let mut audio_bytes = Vec::new();
                loop {
                    match field.chunk().await {
                        Ok(Some(chunk)) => {
                            audio_bytes.extend_from_slice(&chunk);
                        }
                        Ok(None) => break,
                        Err(e) => {
                            error!("Failed to read audio data: {}", e);
                            return json_response(
                                StatusCode::BAD_REQUEST,
                                &ErrorResponse {
                                    error: format!("Failed to read audio data: {}", e),
                                },
                            );
                        }
                    }
                }
                let data = Bytes::from(audio_bytes);

                // Try to parse as WAV file first
                if let Ok(wav_data) = parse_wav(&data) {
                    audio_data = Some(wav_data.0);
                    sample_rate = wav_data.1;
                } else {
                    // Assume raw PCM data
                    audio_data = Some(parse_raw_pcm(&data));
                }
            }
            "sample_rate" => {
                if let Ok(text) = field.text().await {
                    match text.parse::<u32>() {
                        Ok(rate) if (MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&rate) => {
                            sample_rate = rate;
                        }
                        Ok(_) => {
                            return json_response(
                                StatusCode::BAD_REQUEST,
                                &ErrorResponse {
                                    error: format!(
                                        "sample_rate must be between {} and {}",
                                        MIN_SAMPLE_RATE, MAX_SAMPLE_RATE
                                    ),
                                },
                            );
                        }
                        Err(_) => {
                            return json_response(
                                StatusCode::BAD_REQUEST,
                                &ErrorResponse {
                                    error: "sample_rate must be a valid integer".to_string(),
                                },
                            );
                        }
                    }
                }
            }
            "filename" => {
                if let Ok(text) = field.text().await {
                    filename = Some(text);
                }
            }
            _ => {}
        }
    }

    let audio_data = match audio_data {
        Some(data) => data,
        None => {
            return json_response(
                StatusCode::BAD_REQUEST,
                &ErrorResponse {
                    error: "No audio data provided".to_string(),
                },
            );
        }
    };
    if audio_data.is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            &ErrorResponse {
                error: "Audio data is empty".to_string(),
            },
        );
    }

    // Generate filename if not provided
    let filename_raw = filename.unwrap_or_else(|| {
        format!(
            "audio_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        )
    });
    let filename = match validate_filename(&filename_raw) {
        Ok(valid) => valid,
        Err(e) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                &ErrorResponse {
                    error: format!("Invalid filename: {}", e),
                },
            );
        }
    };

    info!(
        "Processing upload: filename={}, samples={}, sample_rate={}Hz",
        filename,
        audio_data.len(),
        sample_rate
    );

    // Resample to 16kHz if needed
    let audio_data = if sample_rate != 16000 {
        resample(&audio_data, sample_rate, 16000)
    } else {
        audio_data
    };
    if audio_data.is_empty() {
        return json_response(
            StatusCode::BAD_REQUEST,
            &ErrorResponse {
                error: "Audio data is empty after preprocessing".to_string(),
            },
        );
    }
    let max_audio_samples = max_audio_samples();
    if audio_data.len() > max_audio_samples {
        return json_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            &ErrorResponse {
                error: format!(
                    "Audio is too long (max {} samples at 16kHz)",
                    max_audio_samples
                ),
            },
        );
    }

    // Save audio file to disk
    let audio_path = std::path::PathBuf::from("data").join(format!("{}.wav", filename));
    if let Err(e) = save_audio_as_wav(&audio_data, &audio_path) {
        error!("Failed to save audio file: {}", e);
        return json_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &ErrorResponse {
                error: format!("Failed to save audio file: {}", e),
            },
        );
    }

    // Create job
    if let Err(e) = service.create_job(filename.clone()).await {
        error!("Failed to create job: {}", e);
        return json_response(
            StatusCode::CONFLICT,
            &ErrorResponse {
                error: format!("Failed to create job: {}", e),
            },
        );
    }

    // Start transcription in background
    service
        .start_transcription(filename.clone(), audio_data)
        .await;

    // Return filename for polling
    let response = nojson::json(|f| {
        f.object(|f| {
            f.member("filename", &filename)?;
            f.member("message", "Transcription started")
        })
    });

    (
        StatusCode::ACCEPTED,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        response.to_string(),
    )
        .into_response()
}

fn json_response<T: nojson::DisplayJson>(status: StatusCode, data: &T) -> Response {
    let json = nojson::json(|f| {
        f.set_spacing(true);
        f.set_indent_size(2);
        f.value(data)
    });

    (
        status,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        json.to_string(),
    )
        .into_response()
}

fn max_audio_samples() -> usize {
    static MAX_AUDIO_SAMPLES: OnceLock<usize> = OnceLock::new();

    *MAX_AUDIO_SAMPLES.get_or_init(|| match std::env::var(MAX_AUDIO_SAMPLES_ENV) {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(value) if value > 0 => value,
            Ok(_) => {
                warn!(
                    "{} must be greater than 0. Using default: {}",
                    MAX_AUDIO_SAMPLES_ENV, DEFAULT_MAX_AUDIO_SAMPLES
                );
                DEFAULT_MAX_AUDIO_SAMPLES
            }
            Err(_) => {
                warn!(
                    "Invalid {}='{}'. Using default: {}",
                    MAX_AUDIO_SAMPLES_ENV, raw, DEFAULT_MAX_AUDIO_SAMPLES
                );
                DEFAULT_MAX_AUDIO_SAMPLES
            }
        },
        Err(_) => DEFAULT_MAX_AUDIO_SAMPLES,
    })
}

fn parse_wav(data: &[u8]) -> Result<(Vec<f32>, u32), String> {
    let cursor = std::io::Cursor::new(data);
    let reader =
        hound::WavReader::new(cursor).map_err(|e| format!("Failed to parse WAV: {}", e))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        hound::SampleFormat::Int => {
            let max_value = 2_i32.pow(spec.bits_per_sample as u32 - 1) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_value)
                .collect()
        }
    };

    Ok((samples, sample_rate))
}

fn parse_raw_pcm(data: &[u8]) -> Vec<f32> {
    let mut samples = Vec::new();
    for chunk in data.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(sample as f32 / 32768.0);
    }
    samples
}

fn resample(data: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if data.is_empty() || from_rate == to_rate {
        return data.to_vec();
    }
    if from_rate == 0 || to_rate == 0 {
        return Vec::new();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (data.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let src_idx_floor = src_idx.floor() as usize;
        let src_idx_ceil = (src_idx_floor + 1).min(data.len() - 1);
        let t = src_idx - src_idx_floor as f64;

        let sample = data[src_idx_floor] * (1.0 - t) as f32 + data[src_idx_ceil] * t as f32;
        output.push(sample);
    }

    output
}

fn save_audio_as_wav(audio_data: &[f32], path: &std::path::PathBuf) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "Invalid output path".to_string())?;
    std::fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| format!("Failed to create WAV writer: {}", e))?;

    for &sample in audio_data {
        let sample_i16 = (sample * 32767.0) as i16;
        writer
            .write_sample(sample_i16)
            .map_err(|e| format!("Failed to write sample: {}", e))?;
    }

    writer
        .finalize()
        .map_err(|e| format!("Failed to finalize WAV: {}", e))?;

    Ok(())
}
