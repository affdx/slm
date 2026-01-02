/**
 * MediaPipe Holistic landmark extraction for browser.
 * 
 * Uses @mediapipe/holistic package which matches the Python baseline's
 * MediaPipe Holistic API for consistent feature extraction.
 *
 * Feature structure (258 total):
 * - Pose landmarks: 33 x 4 (x, y, z, visibility) = 132 features
 * - Left hand landmarks: 21 x 3 (x, y, z) = 63 features
 * - Right hand landmarks: 21 x 3 (x, y, z) = 63 features
 */

import { Holistic, Results } from "@mediapipe/holistic";

// Feature dimensions (must match Python training)
const NUM_POSE_LANDMARKS = 33;
const NUM_HAND_LANDMARKS = 21;
const POSE_FEATURES = NUM_POSE_LANDMARKS * 4; // x, y, z, visibility = 132
const HAND_FEATURES = NUM_HAND_LANDMARKS * 3; // x, y, z = 63
const TOTAL_FEATURES = POSE_FEATURES + HAND_FEATURES * 2; // 132 + 63 + 63 = 258
const NUM_FRAMES = 30;

// Singleton Holistic instance
let holistic: Holistic | null = null;
let initPromise: Promise<void> | null = null;
let lastResults: Results | null = null;

/**
 * Initialize MediaPipe Holistic
 */
async function initializeHolistic(): Promise<void> {
  console.log("[MediaPipe] Initializing Holistic...");
  const startTime = performance.now();

  holistic = new Holistic({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`;
    },
  });

  holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  // Set up results callback
  holistic.onResults((results) => {
    lastResults = results;
  });

  // Initialize by sending a dummy frame
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 480;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    await holistic.send({ image: canvas });
  }

  const loadTime = performance.now() - startTime;
  console.log(`[MediaPipe] Holistic initialized in ${loadTime.toFixed(0)}ms`);
}

/**
 * Ensure Holistic is initialized (singleton pattern)
 */
export async function ensureLandmarkersReady(): Promise<void> {
  if (!initPromise) {
    initPromise = initializeHolistic();
  }
  await initPromise;
}

/**
 * Extract 258 features from Holistic results (matching Python baseline exactly)
 */
function extractKeypoints258(results: Results): Float32Array {
  const features = new Float32Array(TOTAL_FEATURES);

  // Pose landmarks (33 x 4 = 132 features)
  if (results.poseLandmarks) {
    for (let i = 0; i < NUM_POSE_LANDMARKS; i++) {
      const lm = results.poseLandmarks[i];
      if (lm) {
        features[i * 4] = lm.x;
        features[i * 4 + 1] = lm.y;
        features[i * 4 + 2] = lm.z;
        features[i * 4 + 3] = lm.visibility ?? 1.0;
      }
    }
  }

  // Left hand landmarks (21 x 3 = 63 features)
  if (results.leftHandLandmarks) {
    for (let i = 0; i < NUM_HAND_LANDMARKS; i++) {
      const lm = results.leftHandLandmarks[i];
      if (lm) {
        const offset = POSE_FEATURES + i * 3;
        features[offset] = lm.x;
        features[offset + 1] = lm.y;
        features[offset + 2] = lm.z;
      }
    }
  }

  // Right hand landmarks (21 x 3 = 63 features)
  if (results.rightHandLandmarks) {
    for (let i = 0; i < NUM_HAND_LANDMARKS; i++) {
      const lm = results.rightHandLandmarks[i];
      if (lm) {
        const offset = POSE_FEATURES + HAND_FEATURES + i * 3;
        features[offset] = lm.x;
        features[offset + 1] = lm.y;
        features[offset + 2] = lm.z;
      }
    }
  }

  return features;
}

/**
 * Process a single frame with Holistic
 */
async function processFrame(canvas: HTMLCanvasElement): Promise<Float32Array> {
  if (!holistic) {
    throw new Error("Holistic not initialized");
  }

  lastResults = null;
  await holistic.send({ image: canvas });

  // Wait for results (callback based)
  let attempts = 0;
  while (!lastResults && attempts < 100) {
    await new Promise((resolve) => setTimeout(resolve, 10));
    attempts++;
  }

  if (!lastResults) {
    console.warn("[MediaPipe] No results after timeout, returning zeros");
    return new Float32Array(TOTAL_FEATURES);
  }

  return extractKeypoints258(lastResults);
}

/**
 * Extract landmarks from a video element using SLIDING WINDOW approach.
 * 
 * This matches the baseline training approach:
 * - Process ALL frames in the video sequentially
 * - Use the LAST N frames for prediction (sliding window)
 *
 * @param video - Video element to process
 * @param numFrames - Number of frames to use for prediction (default 30)
 * @returns Float32Array of shape [numFrames * 258]
 */
export async function extractLandmarksFromVideo(
  video: HTMLVideoElement,
  numFrames: number = NUM_FRAMES
): Promise<Float32Array> {
  await ensureLandmarkersReady();

  if (!holistic) {
    throw new Error("Holistic not initialized");
  }

  const duration = video.duration;
  if (!duration || duration === 0) {
    throw new Error("Invalid video duration");
  }

  const estimatedFps = 30;
  const estimatedTotalFrames = Math.ceil(duration * estimatedFps);

  console.log(
    `[MediaPipe] Processing video: ${duration.toFixed(2)}s, ~${estimatedTotalFrames} frames (sliding window, last ${numFrames})`
  );
  const startTime = performance.now();

  // Create canvas for frame extraction
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to create canvas context");
  }

  // Process ALL frames and store landmarks
  const allFrameLandmarks: Float32Array[] = [];
  const frameInterval = 1 / estimatedFps;

  for (let time = 0; time < duration; time += frameInterval) {
    // Seek to frame time
    video.currentTime = time;
    await new Promise<void>((resolve) => {
      const onSeeked = () => {
        video.removeEventListener("seeked", onSeeked);
        resolve();
      };
      video.addEventListener("seeked", onSeeked);
    });

    // Draw frame to canvas
    ctx.drawImage(video, 0, 0);

    // Process with Holistic
    const landmarks = await processFrame(canvas);
    allFrameLandmarks.push(landmarks);
  }

  console.log(`[MediaPipe] Processed ${allFrameLandmarks.length} total frames`);

  // Use LAST N frames (sliding window approach matching baseline)
  const result = new Float32Array(numFrames * TOTAL_FEATURES);

  if (allFrameLandmarks.length >= numFrames) {
    const startIdx = allFrameLandmarks.length - numFrames;
    for (let i = 0; i < numFrames; i++) {
      result.set(allFrameLandmarks[startIdx + i], i * TOTAL_FEATURES);
    }
  } else {
    // Video too short - pad with zeros at beginning
    const padding = numFrames - allFrameLandmarks.length;
    for (let i = 0; i < allFrameLandmarks.length; i++) {
      result.set(allFrameLandmarks[i], (padding + i) * TOTAL_FEATURES);
    }
  }

  const extractionTime = performance.now() - startTime;
  console.log(
    `[MediaPipe] Extraction completed in ${extractionTime.toFixed(0)}ms`
  );

  return result;
}

/**
 * Extract ALL frame landmarks from a video (for sliding window inference).
 * 
 * Returns an array of landmarks for each frame, which can be used to
 * run inference on multiple sliding windows and pick the best result.
 *
 * @param video - Video element to process
 * @returns Array of Float32Array, one per frame (each with 258 features)
 */
export async function extractAllFrameLandmarks(
  video: HTMLVideoElement
): Promise<Float32Array[]> {
  await ensureLandmarkersReady();

  if (!holistic) {
    throw new Error("Holistic not initialized");
  }

  const duration = video.duration;
  if (!duration || duration === 0) {
    throw new Error("Invalid video duration");
  }

  const estimatedFps = 30;
  console.log(
    `[MediaPipe] Extracting ALL frame landmarks from video: ${duration.toFixed(2)}s`
  );
  const startTime = performance.now();

  // Create canvas for frame extraction
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to create canvas context");
  }

  // Process ALL frames and store landmarks
  const allFrameLandmarks: Float32Array[] = [];
  const frameInterval = 1 / estimatedFps;

  for (let time = 0; time < duration; time += frameInterval) {
    // Seek to frame time
    video.currentTime = time;
    await new Promise<void>((resolve) => {
      const onSeeked = () => {
        video.removeEventListener("seeked", onSeeked);
        resolve();
      };
      video.addEventListener("seeked", onSeeked);
    });

    // Draw frame to canvas
    ctx.drawImage(video, 0, 0);

    // Process with Holistic
    const landmarks = await processFrame(canvas);
    allFrameLandmarks.push(landmarks);
  }

  const extractionTime = performance.now() - startTime;
  console.log(
    `[MediaPipe] Extracted ${allFrameLandmarks.length} frames in ${extractionTime.toFixed(0)}ms`
  );

  return allFrameLandmarks;
}

/**
 * Extract ALL frame landmarks from a video blob
 */
export async function extractAllFrameLandmarksFromBlob(
  blob: Blob
): Promise<Float32Array[]> {
  const video = document.createElement("video");
  video.playsInline = true;
  video.muted = true;

  const url = URL.createObjectURL(blob);
  video.src = url;

  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error("Failed to load video"));
  });

  try {
    return await extractAllFrameLandmarks(video);
  } finally {
    URL.revokeObjectURL(url);
  }
}

/**
 * Extract landmarks from a video blob
 */
export async function extractLandmarksFromBlob(
  blob: Blob,
  numFrames: number = NUM_FRAMES
): Promise<Float32Array> {
  const video = document.createElement("video");
  video.playsInline = true;
  video.muted = true;

  const url = URL.createObjectURL(blob);
  video.src = url;

  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error("Failed to load video"));
  });

  try {
    return await extractLandmarksFromVideo(video, numFrames);
  } finally {
    URL.revokeObjectURL(url);
  }
}

/**
 * Extract landmarks from a single canvas frame (for real-time processing)
 */
export async function extractLandmarksFromFrame(
  canvas: HTMLCanvasElement
): Promise<Float32Array> {
  await ensureLandmarkersReady();
  return processFrame(canvas);
}

/**
 * Check if Holistic is ready
 */
export function areLandmarkersReady(): boolean {
  return holistic !== null;
}

/**
 * Preload Holistic
 */
export async function preloadLandmarkers(): Promise<void> {
  await ensureLandmarkersReady();
  console.log("[MediaPipe] Holistic preloaded");
}

/**
 * Get feature dimensions info
 */
export function getFeatureDimensions() {
  return {
    numFrames: NUM_FRAMES,
    numFeatures: TOTAL_FEATURES,
    poseFeatures: POSE_FEATURES,
    handFeatures: HAND_FEATURES,
    totalLength: NUM_FRAMES * TOTAL_FEATURES,
  };
}

// Legacy exports for compatibility
export function getCurrentDelegate(): "GPU" | "CPU" {
  return "CPU"; // Holistic doesn't have GPU/CPU toggle
}

export async function setDelegate(_delegate: "GPU" | "CPU"): Promise<void> {
  // No-op for Holistic
}

export async function toggleDelegate(): Promise<"GPU" | "CPU"> {
  return "CPU";
}

/**
 * Landmark point for drawing
 */
export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

/**
 * Drawing data for visualization
 */
export interface DrawingData {
  pose: LandmarkPoint[] | null;
  leftHand: LandmarkPoint[] | null;
  rightHand: LandmarkPoint[] | null;
}

/**
 * Result from landmark extraction with drawing data
 */
export interface LandmarkExtractionResult {
  features: Float32Array;
  drawingData: DrawingData;
}

// Pose connections for drawing
export const POSE_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10], [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];

// Hand connections for drawing
export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

/**
 * Extract landmarks with drawing data (for VideoInferencePlayer)
 */
export async function extractLandmarksWithDrawingData(
  canvas: HTMLCanvasElement
): Promise<LandmarkExtractionResult> {
  await ensureLandmarkersReady();

  if (!holistic) {
    throw new Error("Holistic not initialized");
  }

  lastResults = null;
  await holistic.send({ image: canvas });

  // Wait for results
  let attempts = 0;
  while (!lastResults && attempts < 100) {
    await new Promise((resolve) => setTimeout(resolve, 10));
    attempts++;
  }

  // Use type assertion to help TypeScript - Results type from @mediapipe/holistic
  // declares properties as required but they may actually be undefined at runtime
  const currentResults = lastResults as Results | null;
  const features = currentResults ? extractKeypoints258(currentResults) : new Float32Array(TOTAL_FEATURES);

  const drawingData: DrawingData = {
    pose: null,
    leftHand: null,
    rightHand: null,
  };

  if (currentResults) {
    // Access properties with optional chaining since they may be undefined at runtime
    const pose = currentResults.poseLandmarks;
    const leftHand = currentResults.leftHandLandmarks;
    const rightHand = currentResults.rightHandLandmarks;

    if (pose) {
      drawingData.pose = pose.map((lm) => ({
        x: lm.x,
        y: lm.y,
        z: lm.z,
        visibility: lm.visibility,
      }));
    }
    if (leftHand) {
      drawingData.leftHand = leftHand.map((lm) => ({
        x: lm.x,
        y: lm.y,
        z: lm.z,
      }));
    }
    if (rightHand) {
      drawingData.rightHand = rightHand.map((lm) => ({
        x: lm.x,
        y: lm.y,
        z: lm.z,
      }));
    }
  }

  return { features, drawingData };
}

/**
 * Draw landmarks on canvas
 */
export function drawLandmarks(
  ctx: CanvasRenderingContext2D,
  drawingData: DrawingData,
  width: number,
  height: number,
  options: {
    showPose?: boolean;
    showHands?: boolean;
    poseColor?: string;
    leftHandColor?: string;
    rightHandColor?: string;
    lineWidth?: number;
    pointRadius?: number;
  } = {}
): void {
  const {
    showPose = true,
    showHands = true,
    poseColor = "#00FF00",
    leftHandColor = "#FF0000",
    rightHandColor = "#0000FF",
    lineWidth = 2,
    pointRadius = 4,
  } = options;

  const drawConnections = (
    landmarks: LandmarkPoint[],
    connections: [number, number][],
    color: string
  ) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;

    for (const [start, end] of connections) {
      if (landmarks[start] && landmarks[end]) {
        ctx.beginPath();
        ctx.moveTo(landmarks[start].x * width, landmarks[start].y * height);
        ctx.lineTo(landmarks[end].x * width, landmarks[end].y * height);
        ctx.stroke();
      }
    }
  };

  const drawPoints = (landmarks: LandmarkPoint[], color: string) => {
    ctx.fillStyle = color;
    for (const lm of landmarks) {
      ctx.beginPath();
      ctx.arc(lm.x * width, lm.y * height, pointRadius, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  if (showPose && drawingData.pose) {
    drawConnections(drawingData.pose, POSE_CONNECTIONS, poseColor);
    drawPoints(drawingData.pose, poseColor);
  }

  if (showHands) {
    if (drawingData.leftHand) {
      drawConnections(drawingData.leftHand, HAND_CONNECTIONS, leftHandColor);
      drawPoints(drawingData.leftHand, leftHandColor);
    }
    if (drawingData.rightHand) {
      drawConnections(drawingData.rightHand, HAND_CONNECTIONS, rightHandColor);
      drawPoints(drawingData.rightHand, rightHandColor);
    }
  }
}
