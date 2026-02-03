use crate::models::ErrorResponse;
use crate::services::TranscriptionService;
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use std::sync::Arc;
use tracing::{error, info};

pub async fn upload_handler(
    State(service): State<Arc<TranscriptionService>>,
    mut multipart: Multipart,
) -> Response {
    info!("Received upload request");

    let mut audio_data: Option<Vec<f32>> = None;
    let mut sample_rate: u32 = 16000;
    let mut filename: Option<String> = None;

    // Parse multipart form data
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().map(|s| s.to_string()).unwrap_or_default();

        match name.as_str() {
            "audio" => {
                // Get original filename if available
                if let Some(fname) = field.file_name() {
                    filename = Some(fname.to_string());
                }

                let data: Bytes = match field.bytes().await {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        error!("Failed to read audio data: {}", e);
                        return json_response(
                            StatusCode::BAD_REQUEST,
                            &ErrorResponse {
                                error: format!("Failed to read audio data: {}", e),
                            },
                        );
                    }
                };

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
                    sample_rate = text.parse::<u32>().unwrap_or(16000);
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

    // Generate filename if not provided
    let filename = filename.unwrap_or_else(|| {
        format!(
            "audio_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        )
    });

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
    if from_rate == to_rate {
        return data.to_vec();
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
    std::fs::create_dir_all(path.parent().unwrap())
        .map_err(|e| format!("Failed to create directory: {}", e))?;

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
