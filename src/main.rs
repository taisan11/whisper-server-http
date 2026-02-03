mod handlers;
mod models;
mod services;

use axum::{Router, routing::get, routing::post};
use services::{TranscriptionService, VadService};
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::info;
use whisper_rs::{WhisperContext, WhisperContextParameters};

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    info!("起動中...");

    // Get port from environment variable or use default
    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(3000);

    // Initialize Whisper model
    let model_path = std::env::var("WHISPER_MODEL_PATH").unwrap_or_else(|_| {
        let default_path = "./models/ggml-base.bin";
        info!("モデルが指定されてないからデフォルトパスから読み込むよ");
        default_path.to_string()
    });

    info!("使用するWhisperモデル: {}", model_path);

    let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
        .unwrap_or_else(|e| {
            tracing::error!(
                "Whisperモデルを読み込めなかったよ〜... '{}': {}",
                model_path,
                e
            );
            tracing::error!("モデルのパスを確認してね！");
            tracing::error!("もしモデルがなかったらダウンロードしよう!!");
            std::process::exit(1);
        });

    info!("Whisper読み込み成功!!");
    info!("");

    // Initialize VAD
    let vad_model_path = std::env::var("VAD_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./models/silero_vad.onnx"));

    // Check if VAD model exists
    let vad_service = if vad_model_path.exists() {
        match VadService::new(&vad_model_path, 0.5) {
            Ok(vad) => {
                info!("VADを読み込んだよ");
                Some(vad)
            }
            Err(e) => {
                tracing::warn!("VADの読み込みに失敗したよ...: {}", e);
                tracing::warn!("よって発話時間検出がなくなるよ");
                None
            }
        }
    } else {
        tracing::warn!("VADモデルが見つからなかったよ");
        tracing::warn!("よって発話時間検出がなくなるよ");
        None
    };

    // Create data directory
    let data_dir = PathBuf::from("data");
    std::fs::create_dir_all(&data_dir).expect("Failed to create data directory");

    // Initialize transcription service
    let mut service = TranscriptionService::new(ctx, data_dir);

    // Add VAD if available
    if let Some(vad) = vad_service {
        info!("Enabling VAD for transcription service");
        service = service.with_vad(vad);
    }

    let service = Arc::new(service);

    // Build router
    let app = Router::new()
        .route("/", get(health_check))
        .route("/upload", post(handlers::upload_handler))
        .route("/status", get(handlers::status_handler))
        .route("/finish", get(handlers::finish_handler))
        .layer(CorsLayer::permissive())
        .with_state(service);

    // Start server
    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| {
            tracing::error!("ポート{}にバインドできなかったよ: {}", port, e);
            std::process::exit(1);
        });

    info!("Server listening on {}", addr);
    info!("");
    info!("サーバーが起動したぜ!!");
    info!("================================");
    info!("環境変数:");
    info!("  PORT               - サーバーポート (current: {})", port);
    info!(
        "  WHISPER_MODEL_PATH - Whisperモデルパス (current: {})",
        model_path
    );
    info!(
        "  VAD_MODEL_PATH     - VADモデルパス (current: {})",
        vad_model_path.display()
    );
    info!("  RUST_LOG           - ログレベル");
    info!("================================");

    axum::serve(listener, app)
        .await
        .expect("Server failed to start");
}

async fn health_check() -> &'static str {
    "Whisper HTTP Server is running"
}
