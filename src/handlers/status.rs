use crate::models::ErrorResponse;
use crate::services::TranscriptionService;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use std::sync::Arc;

pub async fn status_handler(
    State(service): State<Arc<TranscriptionService>>,
    axum::extract::Query(query): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let filename = match query.get("filename") {
        Some(f) => f.clone(),
        None => {
            let json = nojson::json(|f| {
                f.set_spacing(true);
                f.set_indent_size(2);
                f.value(&ErrorResponse {
                    error: "Missing 'filename' query parameter".to_string(),
                })
            });

            return (
                StatusCode::BAD_REQUEST,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                json.to_string(),
            )
                .into_response();
        }
    };

    match service.get_job_status(&filename).await {
        Some(job) => {
            let json = nojson::json(|f| {
                f.set_spacing(true);
                f.set_indent_size(2);
                f.value(&job)
            });

            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                json.to_string(),
            )
                .into_response()
        }
        None => {
            let json = nojson::json(|f| {
                f.set_spacing(true);
                f.set_indent_size(2);
                f.value(&ErrorResponse {
                    error: format!("Job not found: {}", filename),
                })
            });

            (
                StatusCode::NOT_FOUND,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                json.to_string(),
            )
                .into_response()
        }
    }
}
