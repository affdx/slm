/**
 * MediaPipe Tasks Vision landmark extraction for browser.
 *
 * Extracts pose and hand landmarks from video frames using
 * MediaPipe Pose Landmarker + Hand Landmarker.
 *
 * Feature structure (258 total):
 * - Pose landmarks: 33 x 4 (x, y, z, visibility) = 132 features
 * - Left hand landmarks: 21 x 3 (x, y, z) = 63 features
 * - Right hand landmarks: 21 x 3 (x, y, z) = 63 features
 *
 * Uses dynamic imports to avoid SSR issues.
 */

// Type imports only (no runtime import)
type PoseLandmarker = import("@mediapipe/tasks-vision").PoseLandmarker;
type HandLandmarker = import("@mediapipe/tasks-vision").HandLandmarker;
type PoseLandmarkerResult =
  import("@mediapipe/tasks-vision").PoseLandmarkerResult;
type HandLandmarkerResult =
  import("@mediapipe/tasks-vision").HandLandmarkerResult;

// Feature dimensions (must match Python training)
const NUM_POSE_LANDMARKS = 33;
const NUM_HAND_LANDMARKS = 21;
const POSE_FEATURES = NUM_POSE_LANDMARKS * 4; // x, y, z, visibility = 132
const HAND_FEATURES = NUM_HAND_LANDMARKS * 3; // x, y, z = 63
const TOTAL_FEATURES = POSE_FEATURES + HAND_FEATURES * 2; // 132 + 63 + 63 = 258
const NUM_FRAMES = 30;

// Singleton instances
let poseLandmarker: PoseLandmarker | null = null;
let handLandmarker: HandLandmarker | null = null;
let initPromise: Promise<void> | null = null;
let currentDelegate: "GPU" | "CPU" = "GPU";

// LocalStorage key for delegate preference
const DELEGATE_STORAGE_KEY = "mediapipe-delegate";

/**
 * Get saved delegate preference from localStorage
 */
function getSavedDelegate(): "GPU" | "CPU" {
  if (typeof localStorage === "undefined") return "GPU";
  const saved = localStorage.getItem(DELEGATE_STORAGE_KEY);
  return saved === "CPU" ? "CPU" : "GPU";
}

/**
 * Save delegate preference to localStorage
 */
function saveDelegate(delegate: "GPU" | "CPU"): void {
  if (typeof localStorage === "undefined") return;
  localStorage.setItem(DELEGATE_STORAGE_KEY, delegate);
}

/**
 * Initialize MediaPipe landmarkers with specified delegate
 */
async function initializeLandmarkers(delegate?: "GPU" | "CPU"): Promise<void> {
  const targetDelegate = delegate ?? getSavedDelegate();
  
  // If already initialized with same delegate, skip
  if (poseLandmarker && handLandmarker && currentDelegate === targetDelegate) {
    return;
  }

  // If switching delegates, close existing landmarkers
  if (poseLandmarker || handLandmarker) {
    console.log(`[MediaPipe] Switching delegate from ${currentDelegate} to ${targetDelegate}`);
    if (poseLandmarker) {
      poseLandmarker.close();
      poseLandmarker = null;
    }
    if (handLandmarker) {
      handLandmarker.close();
      handLandmarker = null;
    }
    initPromise = null;
  }

  console.log(`[MediaPipe] Initializing landmarkers with ${targetDelegate} delegate...`);
  const startTime = performance.now();

  // Dynamic import
  const { PoseLandmarker, HandLandmarker, FilesetResolver } = await import(
    "@mediapipe/tasks-vision"
  );

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  // Initialize Pose Landmarker
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: targetDelegate,
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  // Initialize Hand Landmarker
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: targetDelegate,
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  currentDelegate = targetDelegate;
  saveDelegate(targetDelegate);

  const loadTime = performance.now() - startTime;
  console.log(
    `[MediaPipe] Landmarkers initialized with ${targetDelegate} delegate in ${loadTime.toFixed(0)}ms`
  );
}

/**
 * Ensure landmarkers are initialized (singleton pattern)
 */
export async function ensureLandmarkersReady(): Promise<void> {
  if (!initPromise) {
    initPromise = initializeLandmarkers();
  }
  await initPromise;
}

/**
 * Extract landmarks from a single frame
 * 
 * IMPORTANT: MediaPipe HandLandmarker returns handedness from the camera's perspective,
 * which is MIRRORED from the person's perspective.
 * - "Left" from camera = person's RIGHT hand
 * - "Right" from camera = person's LEFT hand
 * 
 * However, MediaPipe Holistic (used in Python training) returns from person's perspective.
 * So we need to NOT swap - use the labels directly as the Python code expects.
 */
function extractFrameLandmarks(
  poseResult: PoseLandmarkerResult,
  handResult: HandLandmarkerResult
): Float32Array {
  const features = new Float32Array(TOTAL_FEATURES);

  // Extract pose landmarks (33 x 4 = 132 features)
  // Use worldLandmarks if available for visibility, otherwise use landmarks
  if (poseResult.landmarks && poseResult.landmarks.length > 0) {
    const poseLandmarks = poseResult.landmarks[0];
    // worldLandmarks have visibility property
    const worldLandmarks = poseResult.worldLandmarks?.[0];
    
    for (let i = 0; i < NUM_POSE_LANDMARKS; i++) {
      const landmark = poseLandmarks[i];
      if (landmark) {
        features[i * 4] = landmark.x;
        features[i * 4 + 1] = landmark.y;
        features[i * 4 + 2] = landmark.z;
        // Use visibility from worldLandmarks if available, otherwise default to 1.0
        // Note: Tasks Vision API might have visibility in worldLandmarks
        const worldLm = worldLandmarks?.[i];
        features[i * 4 + 3] = (worldLm as { visibility?: number })?.visibility ?? 1.0;
      }
    }
  }

  // Extract hand landmarks
  // 
  // IMPORTANT: MediaPipe HandLandmarker returns handedness from the CAMERA's perspective,
  // which is MIRRORED from the person's perspective:
  // - "Left" label from HandLandmarker = person's RIGHT hand
  // - "Right" label from HandLandmarker = person's LEFT hand
  // 
  // MediaPipe Holistic (used in baseline training) returns from the PERSON's perspective
  // where left_hand_landmarks = person's left hand.
  // 
  // To match the baseline training data, we MUST SWAP the hand assignments:
  // - HandLandmarker "Left" -> assign to RIGHT hand slot (person's right)
  // - HandLandmarker "Right" -> assign to LEFT hand slot (person's left)
  
  let leftHandIdx = -1;
  let rightHandIdx = -1;

  if (handResult.handedness) {
    for (let i = 0; i < handResult.handedness.length; i++) {
      const handedness = handResult.handedness[i];
      if (handedness && handedness.length > 0) {
        const label = handedness[0].categoryName;
        // SWAP: HandLandmarker labels are from camera perspective (mirrored)
        if (label === "Left") {
          rightHandIdx = i;  // Camera's "Left" = Person's RIGHT hand
        } else if (label === "Right") {
          leftHandIdx = i;   // Camera's "Right" = Person's LEFT hand
        }
      }
    }
  }

  // Left hand landmarks (21 x 3 = 63 features) - person's left hand
  if (leftHandIdx >= 0 && handResult.landmarks[leftHandIdx]) {
    const leftHand = handResult.landmarks[leftHandIdx];
    for (let i = 0; i < NUM_HAND_LANDMARKS; i++) {
      const landmark = leftHand[i];
      if (landmark) {
        const offset = POSE_FEATURES + i * 3;
        features[offset] = landmark.x;
        features[offset + 1] = landmark.y;
        features[offset + 2] = landmark.z;
      }
    }
  }

  // Right hand landmarks (21 x 3 = 63 features) - person's right hand
  if (rightHandIdx >= 0 && handResult.landmarks[rightHandIdx]) {
    const rightHand = handResult.landmarks[rightHandIdx];
    for (let i = 0; i < NUM_HAND_LANDMARKS; i++) {
      const landmark = rightHand[i];
      if (landmark) {
        const offset = POSE_FEATURES + HAND_FEATURES + i * 3;
        features[offset] = landmark.x;
        features[offset + 1] = landmark.y;
        features[offset + 2] = landmark.z;
      }
    }
  }

  return features;
}

/**
 * Extract landmarks from a video element
 *
 * @param video - Video element to process
 * @param numFrames - Number of frames to extract (default 30)
 * @returns Float32Array of shape [numFrames * 258]
 */
export async function extractLandmarksFromVideo(
  video: HTMLVideoElement,
  numFrames: number = NUM_FRAMES
): Promise<Float32Array> {
  await ensureLandmarkersReady();

  if (!poseLandmarker || !handLandmarker) {
    throw new Error("Landmarkers not initialized");
  }

  const duration = video.duration;
  if (!duration || duration === 0) {
    throw new Error("Invalid video duration");
  }

  console.log(
    `[MediaPipe] Extracting ${numFrames} frames from ${duration.toFixed(2)}s video`
  );
  const startTime = performance.now();

  // Calculate frame times to sample (same as Python: np.linspace)
  const frameTimes: number[] = [];
  for (let i = 0; i < numFrames; i++) {
    frameTimes.push((i / (numFrames - 1)) * duration);
  }

  // Create canvas for frame extraction
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to create canvas context");
  }

  // Extract landmarks for each frame
  const allLandmarks = new Float32Array(numFrames * TOTAL_FEATURES);
  let timestampMs = 0;

  for (let i = 0; i < numFrames; i++) {
    // Seek to frame time
    video.currentTime = frameTimes[i];
    await new Promise<void>((resolve) => {
      const onSeeked = () => {
        video.removeEventListener("seeked", onSeeked);
        resolve();
      };
      video.addEventListener("seeked", onSeeked);
    });

    // Draw frame to canvas
    ctx.drawImage(video, 0, 0);

    // Process with MediaPipe - use incrementing timestamp for VIDEO mode
    timestampMs = performance.now();
    const poseResult = poseLandmarker.detectForVideo(canvas, timestampMs);
    const handResult = handLandmarker.detectForVideo(canvas, timestampMs);

    // Extract features
    const frameLandmarks = extractFrameLandmarks(poseResult, handResult);
    allLandmarks.set(frameLandmarks, i * TOTAL_FEATURES);
  }

  const extractionTime = performance.now() - startTime;
  console.log(
    `[MediaPipe] Landmark extraction completed in ${extractionTime.toFixed(0)}ms`
  );

  return allLandmarks;
}

/**
 * Extract landmarks from a video blob
 */
export async function extractLandmarksFromBlob(
  blob: Blob,
  numFrames: number = NUM_FRAMES
): Promise<Float32Array> {
  // Create video element
  const video = document.createElement("video");
  video.playsInline = true;
  video.muted = true;

  // Load video from blob
  const url = URL.createObjectURL(blob);
  video.src = url;

  // Wait for video to load metadata
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
 * Process a single image/frame and return landmarks
 */
export async function extractLandmarksFromImage(
  image: HTMLImageElement | HTMLCanvasElement
): Promise<Float32Array> {
  await ensureLandmarkersReady();

  if (!poseLandmarker || !handLandmarker) {
    throw new Error("Landmarkers not initialized");
  }

  // For single image, use IMAGE mode results
  // We need to temporarily switch to IMAGE mode
  const canvas = document.createElement("canvas");
  if (image instanceof HTMLImageElement) {
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
  } else {
    canvas.width = image.width;
    canvas.height = image.height;
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to create canvas context");
  }
  ctx.drawImage(image, 0, 0);

  const timestampMs = performance.now();
  const poseResult = poseLandmarker.detectForVideo(canvas, timestampMs);
  const handResult = handLandmarker.detectForVideo(canvas, timestampMs);

  return extractFrameLandmarks(poseResult, handResult);
}

/**
 * Check if landmarkers are ready
 */
export function areLandmarkersReady(): boolean {
  return poseLandmarker !== null && handLandmarker !== null;
}

/**
 * Preload landmarkers
 */
export async function preloadLandmarkers(): Promise<void> {
  await ensureLandmarkersReady();
  console.log("[MediaPipe] Landmarkers preloaded");
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

/**
 * Get current delegate being used
 */
export function getCurrentDelegate(): "GPU" | "CPU" {
  return currentDelegate;
}

/**
 * Set delegate and reinitialize landmarkers
 * Call this to switch between GPU and CPU processing
 */
export async function setDelegate(delegate: "GPU" | "CPU"): Promise<void> {
  initPromise = initializeLandmarkers(delegate);
  await initPromise;
}

/**
 * Toggle between GPU and CPU delegate
 */
export async function toggleDelegate(): Promise<"GPU" | "CPU"> {
  const newDelegate = currentDelegate === "GPU" ? "CPU" : "GPU";
  await setDelegate(newDelegate);
  return newDelegate;
}

/**
 * Extract landmarks from a single canvas frame (for real-time processing)
 * This is optimized for continuous webcam processing.
 * 
 * @param canvas - Canvas element with current frame drawn
 * @returns Float32Array of 258 features for single frame
 */
export async function extractLandmarksFromFrame(
  canvas: HTMLCanvasElement
): Promise<Float32Array> {
  await ensureLandmarkersReady();

  if (!poseLandmarker || !handLandmarker) {
    throw new Error("Landmarkers not initialized");
  }

  // Use monotonically increasing timestamp for VIDEO mode
  const timestampMs = performance.now();
  const poseResult = poseLandmarker.detectForVideo(canvas, timestampMs);
  const handResult = handLandmarker.detectForVideo(canvas, timestampMs);

  return extractFrameLandmarks(poseResult, handResult);
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

// Pose connections for drawing skeleton lines
export const POSE_CONNECTIONS: [number, number][] = [
  // Face
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
  // Torso
  [9, 10], [11, 12], [11, 23], [12, 24], [23, 24],
  // Left arm
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  // Right arm
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  // Left leg
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  // Right leg
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];

// Hand connections for drawing skeleton lines
export const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm
  [5, 9], [9, 13], [13, 17],
];

/**
 * Extract landmarks from a single canvas frame with drawing data
 * Returns both the feature array for inference AND raw landmark positions for visualization
 * 
 * @param canvas - Canvas element with current frame drawn
 * @returns Object with features array and drawing data
 */
export async function extractLandmarksWithDrawingData(
  canvas: HTMLCanvasElement
): Promise<LandmarkExtractionResult> {
  await ensureLandmarkersReady();

  if (!poseLandmarker || !handLandmarker) {
    throw new Error("Landmarkers not initialized");
  }

  // Use monotonically increasing timestamp for VIDEO mode
  const timestampMs = performance.now();
  const poseResult = poseLandmarker.detectForVideo(canvas, timestampMs);
  const handResult = handLandmarker.detectForVideo(canvas, timestampMs);

  // Extract features for inference
  const features = extractFrameLandmarks(poseResult, handResult);

  // Extract drawing data
  const drawingData: DrawingData = {
    pose: null,
    leftHand: null,
    rightHand: null,
  };

  // Pose landmarks
  if (poseResult.landmarks && poseResult.landmarks.length > 0) {
    const worldLandmarks = poseResult.worldLandmarks?.[0];
    drawingData.pose = poseResult.landmarks[0].map((lm, i) => ({
      x: lm.x,
      y: lm.y,
      z: lm.z,
      visibility: (worldLandmarks?.[i] as { visibility?: number })?.visibility,
    }));
  }

  // Hand landmarks - determine left/right based on handedness
  // SWAP labels: HandLandmarker returns from camera perspective (mirrored)
  // For drawing, we keep the visual position but swap the semantic label
  if (handResult.landmarks && handResult.handedness) {
    for (let i = 0; i < handResult.landmarks.length; i++) {
      const handedness = handResult.handedness[i];
      if (handedness && handedness.length > 0) {
        const label = handedness[0].categoryName;
        const landmarks = handResult.landmarks[i].map((lm) => ({
          x: lm.x,
          y: lm.y,
          z: lm.z,
        }));

        // SWAP: Camera's "Left" = Person's RIGHT, Camera's "Right" = Person's LEFT
        if (label === "Left") {
          drawingData.rightHand = landmarks;  // Person's right hand
        } else if (label === "Right") {
          drawingData.leftHand = landmarks;   // Person's left hand
        }
      }
    }
  }

  return { features, drawingData };
}

/**
 * Draw landmarks on a canvas
 * 
 * @param ctx - Canvas 2D context
 * @param drawingData - Landmark drawing data
 * @param width - Canvas width
 * @param height - Canvas height
 * @param options - Drawing options
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

  // Helper to draw connections
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

  // Helper to draw points
  const drawPoints = (landmarks: LandmarkPoint[], color: string) => {
    ctx.fillStyle = color;

    for (const lm of landmarks) {
      ctx.beginPath();
      ctx.arc(lm.x * width, lm.y * height, pointRadius, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  // Draw pose
  if (showPose && drawingData.pose) {
    drawConnections(drawingData.pose, POSE_CONNECTIONS, poseColor);
    drawPoints(drawingData.pose, poseColor);
  }

  // Draw hands
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
