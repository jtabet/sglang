use axum::{
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

#[derive(Serialize)]
struct ErrorResponse<'a> {
    error: ErrorDetail<'a>,
}

#[derive(Serialize)]
struct ErrorDetail<'a> {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'a str,
    message: &'a str,
}

pub const HEADER_X_SMG_ERROR_CODE: &str = "X-SMG-Error-Code";

pub fn internal_error(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::INTERNAL_SERVER_ERROR, code, message)
}

pub fn bad_request(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::BAD_REQUEST, code, message)
}

pub fn not_found(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::NOT_FOUND, code, message)
}

pub fn service_unavailable(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::SERVICE_UNAVAILABLE, code, message)
}

/// 425 Too Early (RFC 9163 §2 — originally RFC 8470): the server is unwilling
/// to process the request yet because the state it depends on is not ready.
///
/// Used by TTFT budget gating: the prompt cache for this request is cold (the
/// cache-aware policy's match_rate predicts a first token slower than the
/// caller's `X-TTFT-Budget-Ms` budget), so the gateway declines immediately
/// — typically to let a router re-select a warmer worker — while the prefill
/// itself keeps running in the background so the cache is warm for the next
/// request. Semantics deliberately differ from 503: the workers are healthy,
/// nothing is overloaded, and retrying the same request right now would hit
/// the same estimate; the caller should route elsewhere or wait for the
/// warmup to land.
///
/// `code` should be `prefill_budget_exceeded`; the body may carry the
/// estimate details for the caller's logs.
pub fn too_early(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::TOO_EARLY, code, message)
}

pub fn failed_dependency(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::FAILED_DEPENDENCY, code, message)
}

pub fn not_implemented(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::NOT_IMPLEMENTED, code, message)
}

pub fn bad_gateway(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::BAD_GATEWAY, code, message)
}

pub fn method_not_allowed(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::METHOD_NOT_ALLOWED, code, message)
}

pub fn create_error(
    status: StatusCode,
    code: impl Into<String>,
    message: impl Into<String>,
) -> Response {
    let code_str = code.into();
    let message_str = message.into();

    let mut headers = HeaderMap::with_capacity(1);
    headers.insert(
        HEADER_X_SMG_ERROR_CODE,
        HeaderValue::from_str(&code_str).unwrap(),
    );

    (
        status,
        headers,
        Json(ErrorResponse {
            error: ErrorDetail {
                error_type: status_code_to_str(status),
                code: &code_str,
                message: &message_str,
            },
        }),
    )
        .into_response()
}

fn status_code_to_str(status_code: StatusCode) -> &'static str {
    status_code
        .canonical_reason()
        .unwrap_or("Unknown Status Code")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    #[tokio::test]
    async fn too_early_response_shape() {
        let res = too_early("prefill_budget_exceeded", "est ttft 84000ms > budget 10000ms");
        assert_eq!(res.status(), StatusCode::TOO_EARLY);
        assert_eq!(
            res.headers()
                .get(HEADER_X_SMG_ERROR_CODE)
                .and_then(|v| v.to_str().ok()),
            Some("prefill_budget_exceeded")
        );
        let (_, body) = res.into_parts();
        let bytes = to_bytes(body, usize::MAX).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains(r#""type":"Too Early""#), "body: {}", text);
        assert!(text.contains("prefill_budget_exceeded"), "body: {}", text);
    }
}

pub fn extract_error_code_from_response<B>(response: &Response<B>) -> &str {
    response
        .headers()
        .get(HEADER_X_SMG_ERROR_CODE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
}
