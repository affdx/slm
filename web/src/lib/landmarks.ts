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

/**
 * Initialize MediaPipe landmarkers
 */
async function initializeLandmarkers(): Promise<void> {
  if (poseLandmarker && handLandmarker) {
    return;
  }

  console.log("[MediaPipe] Initializing landmarkers...");
  const startTime = performance.now();

  // Dynamic import
  const { PoseLandmarker, HandLandmarker, FilesetResolver } = await import(
    "@mediapipe/tasks-vision"
  );

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  // Initialize Pose Landmarker with outputSegmentationMasks to get visibility
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
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
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  const loadTime = performance.now() - startTime;
  console.log(
    `[MediaPipe] Landmarkers initialized in ${loadTime.toFixed(0)}ms`
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
  // MediaPipe HandLandmarker handedness convention:
  // The label indicates which hand it is from the PERSON's perspective
  // (i.e., "Left" = person's left hand, "Right" = person's right hand)
  // 
  // This matches MediaPipe Holistic used in Python training, so we use
  // the labels directly without swapping.
  
  let leftHandIdx = -1;
  let rightHandIdx = -1;

  if (handResult.handedness) {
    for (let i = 0; i < handResult.handedness.length; i++) {
      const handedness = handResult.handedness[i];
      if (handedness && handedness.length > 0) {
        const label = handedness[0].categoryName;
        if (label === "Left") {
          leftHandIdx = i;
        } else if (label === "Right") {
          rightHandIdx = i;
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
