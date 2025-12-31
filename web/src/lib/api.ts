/**
 * API Client for Malaysian Sign Language Translation
 *
 * In production, API calls are proxied through Next.js API routes to:
 * - Hide the backend URL from the client
 * - Enable internal networking on Sevalla
 * - Handle authentication if needed
 *
 * Set USE_PROXY=false for direct backend calls (local development)
 */

// Use proxy routes by default (production), direct calls for local dev if configured
const USE_PROXY = process.env.NEXT_PUBLIC_USE_PROXY !== "false";
const DIRECT_API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// API endpoints - use proxy routes or direct backend
const getApiUrl = (path: string) => {
  if (USE_PROXY) {
    // Use Next.js API routes as proxy
    return `/api${path}`;
  }
  // Direct backend calls (local development)
  return `${DIRECT_API_URL}${path}`;
};

export interface PredictionResult {
  predicted_gloss: string;
  confidence: number;
  top_5_predictions: Array<{
    gloss: string;
    confidence: number;
  }>;
  processing_time_ms: number;
}

export interface GlossInfo {
  gloss: string;
  index: number;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  device: string;
  num_classes: number;
}

export interface ApiError {
  detail: string;
}

/**
 * Predict sign language from video file
 */
export async function predictFromVideo(
  videoFile: File,
  topK: number = 5
): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("video", videoFile);
  formData.append("top_k", topK.toString());

  const response = await fetch(getApiUrl("/predict"), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    const message =
      typeof error.detail === "string"
        ? error.detail
        : "Failed to process video";
    throw new Error(message);
  }

  // Map API response to frontend format
  const data = await response.json();
  return {
    predicted_gloss: data.gloss,
    confidence: data.confidence,
    top_5_predictions: data.top_k.map(
      (item: { gloss: string; confidence: number }) => ({
        gloss: item.gloss,
        confidence: item.confidence,
      })
    ),
    processing_time_ms: data.total_time_ms,
  };
}

/**
 * Predict sign language from pre-extracted landmarks
 * Note: This endpoint is not proxied, uses direct backend call
 */
export async function predictFromLandmarks(
  landmarks: number[][],
  confidenceThreshold: number = 0.3
): Promise<PredictionResult> {
  const response = await fetch(
    `${DIRECT_API_URL}/predict/landmarks?confidence_threshold=${confidenceThreshold}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ landmarks }),
    }
  );

  if (!response.ok) {
    const error: ApiError = await response.json();
    throw new Error(error.detail || "Failed to process landmarks");
  }

  return response.json();
}

/**
 * Get list of all available glosses
 */
export async function getGlosses(): Promise<string[]> {
  const response = await fetch(getApiUrl("/glosses"));

  if (!response.ok) {
    throw new Error("Failed to fetch glosses");
  }

  const data = await response.json();
  return data.glosses;
}

/**
 * Check API health status
 */
export async function getHealthStatus(): Promise<HealthStatus> {
  const response = await fetch(getApiUrl("/health"));

  if (!response.ok) {
    throw new Error("API is not healthy");
  }

  return response.json();
}

/**
 * Format confidence as percentage string
 */
export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`;
}

/**
 * Format gloss name for display (replace underscores with spaces, capitalize)
 */
export function formatGlossName(gloss: string): string {
  return gloss
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
