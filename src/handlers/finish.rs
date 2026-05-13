use super::validate_filename;
use crate::models::ErrorResponse;
use crate::services::TranscriptionService;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use std::sync::Arc;
use tracing::info;

pub async fn finish_handler(
    State(service): State<Arc<TranscriptionService>>,
    axum::extract::Query(query): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let filename = match query.get("filename") {
        Some(f) => match validate_filename(f) {
            Ok(valid) => valid,
            Err(e) => {
                let json = nojson::json(|f| {
                    f.set_spacing(true);
                    f.set_indent_size(2);
                    f.value(&ErrorResponse {
                        error: format!("Invalid 'filename' query parameter: {}", e),
                    })
                });

                return (
                    StatusCode::BAD_REQUEST,
                    [(axum::http::header::CONTENT_TYPE, "application/json")],
                    json.to_string(),
                )
                    .into_response();
            }
        },
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

    info!("Finish request for: {}", filename);

    // Check job status first
    let job = match service.get_job_status(&filename).await {
        Some(job) => job,
        None => {
            let json = nojson::json(|f| {
                f.set_spacing(true);
                f.set_indent_size(2);
                f.value(&ErrorResponse {
                    error: format!("Job not found: {}", filename),
                })
            });

            return (
                StatusCode::NOT_FOUND,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                json.to_string(),
            )
                .into_response();
        }
    };

    // Check if job is completed
    if job.status != crate::models::TranscriptionStatus::Completed {
        let json = nojson::json(|f| {
            f.set_spacing(true);
            f.set_indent_size(2);
            f.object(|f| {
                f.member("error", "Job is not completed yet")?;
                f.member("status", &job.status)?;
                f.member("progress", job.progress)
            })
        });

        return (
            StatusCode::BAD_REQUEST,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            json.to_string(),
        )
            .into_response();
    }

    // Get result
    match service.get_result(&filename).await {
        Some(result) => {
            let json = nojson::json(|f| {
                f.set_spacing(true);
                f.set_indent_size(2);
                f.value(&result)
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
                    error: "Result not found (this should not happen)".to_string(),
                })
            });

            (
                StatusCode::INTERNAL_SERVER_ERROR,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                json.to_string(),
            )
                .into_response()
        }
    }
}
