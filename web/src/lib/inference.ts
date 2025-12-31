/**
 * ONNX Runtime Web inference module for sign language classification.
 *
 * Uses WebGL backend for GPU acceleration in the browser.
 * Uses dynamic imports to avoid SSR issues.
 */

// Model configuration
const MODEL_PATH = "/model.onnx";
const CLASS_MAPPING_PATH = "/class_mapping.json";
const NUM_FRAMES = 30;
const NUM_FEATURES = 258;

// ONNX Runtime types (imported dynamically)
interface OrtSession {
  inputNames: string[];
  outputNames: string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array }>>;
}

interface OrtModule {
  InferenceSession: {
    create(path: string, options?: unknown): Promise<OrtSession>;
  };
  Tensor: new (type: string, data: Float32Array, dims: number[]) => unknown;
  env: {
    wasm: {
      numThreads: number;
      simd: boolean;
      wasmPaths?: string;
    };
  };
}

// Singleton session instance
let sessionPromise: Promise<OrtSession> | null = null;
let classMapping: Record<string, number> | null = null;
let idxToClass: Record<number, string> | null = null;
let ortModule: OrtModule | null = null;

export interface InferenceResult {
  gloss: string;
  confidence: number;
  classId: number;
  topK: Array<{
    gloss: string;
    confidence: number;
    classId: number;
  }>;
  inferenceTimeMs: number;
}

/**
 * Dynamically import ONNX Runtime from CDN (client-side only)
 */
async function getOrt(): Promise<OrtModule> {
  if (ortModule) {
    return ortModule;
  }

  console.log("[ONNX] Loading ONNX Runtime from CDN...");
  
  // Use a stable version that works well with browser
  const cdnUrl = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js";
  
  // Check if already loaded (from script tag)
  if (typeof window !== "undefined" && (window as unknown as { ort: OrtModule }).ort) {
    ortModule = (window as unknown as { ort: OrtModule }).ort;
    console.log("[ONNX] Using existing ONNX Runtime from window");
    return ortModule;
  }
  
  // Dynamically load the script
  await new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = cdnUrl;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load ONNX Runtime from CDN"));
    document.head.appendChild(script);
  });
  
  // Wait a bit for the script to initialize
  await new Promise(resolve => setTimeout(resolve, 100));
  
  if (typeof window !== "undefined" && (window as unknown as { ort: OrtModule }).ort) {
    ortModule = (window as unknown as { ort: OrtModule }).ort;
    
    // Configure ONNX Runtime - use WASM only for reliability
    ortModule.env.wasm.numThreads = 1;
    ortModule.env.wasm.simd = true;
    
    console.log("[ONNX] ONNX Runtime loaded from CDN");
    return ortModule;
  }
  
  throw new Error("ONNX Runtime not available after loading script");
}

/**
 * Load the ONNX model session (singleton pattern)
 */
export async function loadModel(): Promise<OrtSession> {
  if (sessionPromise) {
    return sessionPromise;
  }

  sessionPromise = (async () => {
    const ort = await getOrt();

    console.log("[ONNX] Loading model from:", MODEL_PATH);
    const startTime = performance.now();

    try {
      // Use WASM backend for reliability (WebGL can have issues)
      const session = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });

      const loadTime = performance.now() - startTime;
      console.log(`[ONNX] Model loaded in ${loadTime.toFixed(0)}ms`);
      console.log("[ONNX] Input names:", session.inputNames);
      console.log("[ONNX] Output names:", session.outputNames);

      return session;
    } catch (error) {
      console.error("[ONNX] Failed to load model:", error);
      sessionPromise = null;
      throw error;
    }
  })();

  return sessionPromise;
}

/**
 * Load class mapping from JSON
 */
export async function loadClassMapping(): Promise<Record<string, number>> {
  if (classMapping) {
    return classMapping;
  }

  console.log("[ONNX] Loading class mapping...");
  const response = await fetch(CLASS_MAPPING_PATH);
  if (!response.ok) {
    throw new Error(`Failed to load class mapping: ${response.statusText}`);
  }

  classMapping = await response.json();

  // Create reverse mapping
  idxToClass = {};
  for (const [gloss, idx] of Object.entries(classMapping!)) {
    idxToClass[idx] = gloss;
  }

  console.log(`[ONNX] Loaded ${Object.keys(classMapping!).length} classes`);
  return classMapping!;
}

/**
 * Get gloss name from class index
 */
export function getGlossFromIndex(index: number): string {
  if (!idxToClass) {
    throw new Error("Class mapping not loaded. Call loadClassMapping() first.");
  }
  return idxToClass[index] || "unknown";
}

/**
 * Normalize landmarks (zero mean, unit variance)
 * Matches the Python training normalization
 */
function normalizeLandmarks(landmarks: Float32Array): Float32Array {
  const n = landmarks.length;

  // Calculate mean
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += landmarks[i];
  }
  const mean = sum / n;

  // Calculate std
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const diff = landmarks[i] - mean;
    sumSq += diff * diff;
  }
  const std = Math.sqrt(sumSq / n);

  // Normalize
  const normalized = new Float32Array(n);
  if (std < 1e-6) {
    // Avoid division by zero
    for (let i = 0; i < n; i++) {
      normalized[i] = landmarks[i] - mean;
    }
  } else {
    for (let i = 0; i < n; i++) {
      normalized[i] = (landmarks[i] - mean) / std;
    }
  }

  return normalized;
}

/**
 * Softmax function
 */
function softmax(logits: Float32Array): Float32Array {
  // Find max without spread operator for TypeScript compatibility
  let maxLogit = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }

  const expScores = new Float32Array(logits.length);
  let sumExp = 0;

  for (let i = 0; i < logits.length; i++) {
    expScores[i] = Math.exp(logits[i] - maxLogit);
    sumExp += expScores[i];
  }

  for (let i = 0; i < logits.length; i++) {
    expScores[i] /= sumExp;
  }

  return expScores;
}

/**
 * Run inference on landmarks
 *
 * @param landmarks - Float32Array of shape [30, 258] flattened to [7740]
 * @param topK - Number of top predictions to return
 * @param confidenceThreshold - Minimum confidence to return a prediction
 */
export async function runInference(
  landmarks: Float32Array,
  topK: number = 5,
  confidenceThreshold: number = 0.3
): Promise<InferenceResult> {
  // Validate input
  const expectedLength = NUM_FRAMES * NUM_FEATURES;
  if (landmarks.length !== expectedLength) {
    throw new Error(
      `Invalid landmarks length: expected ${expectedLength}, got ${landmarks.length}`
    );
  }

  // Load model and class mapping if needed
  const [session, ort] = await Promise.all([
    loadModel(),
    loadClassMapping().then(() => getOrt()),
  ]);

  // Normalize landmarks
  const normalizedLandmarks = normalizeLandmarks(landmarks);

  // Create input tensor [1, 30, 258]
  const inputTensor = new ort.Tensor("float32", normalizedLandmarks, [
    1,
    NUM_FRAMES,
    NUM_FEATURES,
  ]);

  // Run inference
  const startTime = performance.now();
  const results = await session.run({ landmarks: inputTensor });
  const inferenceTime = performance.now() - startTime;

  // Get output logits
  const logits = results.logits.data as Float32Array;

  // Apply softmax to get probabilities
  const probs = softmax(logits);

  // Get top-k predictions
  const indices = Array.from({ length: probs.length }, (_, i) => i);
  indices.sort((a, b) => probs[b] - probs[a]);

  const topKResults = indices.slice(0, topK).map((idx) => ({
    gloss: getGlossFromIndex(idx),
    confidence: probs[idx],
    classId: idx,
  }));

  // Get best prediction
  const bestIdx = indices[0];
  const bestConf = probs[bestIdx];
  const bestGloss =
    bestConf >= confidenceThreshold ? getGlossFromIndex(bestIdx) : "unknown";

  console.log(
    `[ONNX] Inference completed in ${inferenceTime.toFixed(1)}ms - ${bestGloss} (${(bestConf * 100).toFixed(1)}%)`
  );

  return {
    gloss: bestGloss,
    confidence: bestConf,
    classId: bestIdx,
    topK: topKResults,
    inferenceTimeMs: inferenceTime,
  };
}

/**
 * Check if the model is loaded
 */
export function isModelLoaded(): boolean {
  return sessionPromise !== null;
}

/**
 * Get list of all glosses
 */
export async function getGlosses(): Promise<string[]> {
  const mapping = await loadClassMapping();
  return Object.keys(mapping).sort();
}

/**
 * Preload the model (call on app init for faster first inference)
 */
export async function preloadModel(): Promise<void> {
  await Promise.all([loadModel(), loadClassMapping()]);
  console.log("[ONNX] Model and class mapping preloaded");
}

/**
 * Run inference with pre-normalized landmarks (for testing/debugging)
 * This skips normalization since the input is already normalized.
 */
export async function runInferenceWithNormalizedLandmarks(
  normalizedLandmarks: Float32Array,
  topK: number = 5,
  confidenceThreshold: number = 0.3
): Promise<InferenceResult> {
  const expectedLength = NUM_FRAMES * NUM_FEATURES;
  if (normalizedLandmarks.length !== expectedLength) {
    throw new Error(
      `Invalid landmarks length: expected ${expectedLength}, got ${normalizedLandmarks.length}`
    );
  }

  const [session, ort] = await Promise.all([
    loadModel(),
    loadClassMapping().then(() => getOrt()),
  ]);

  // Create input tensor [1, 30, 258] - NO normalization
  const inputTensor = new ort.Tensor("float32", normalizedLandmarks, [
    1,
    NUM_FRAMES,
    NUM_FEATURES,
  ]);

  const startTime = performance.now();
  const results = await session.run({ landmarks: inputTensor });
  const inferenceTime = performance.now() - startTime;

  const logits = results.logits.data as Float32Array;
  const probs = softmax(logits);

  const indices = Array.from({ length: probs.length }, (_, i) => i);
  indices.sort((a, b) => probs[b] - probs[a]);

  const topKResults = indices.slice(0, topK).map((idx) => ({
    gloss: getGlossFromIndex(idx),
    confidence: probs[idx],
    classId: idx,
  }));

  const bestIdx = indices[0];
  const bestConf = probs[bestIdx];
  const bestGloss =
    bestConf >= confidenceThreshold ? getGlossFromIndex(bestIdx) : "unknown";

  console.log(
    `[ONNX] Inference (pre-normalized) in ${inferenceTime.toFixed(1)}ms - ${bestGloss} (${(bestConf * 100).toFixed(1)}%)`
  );

  return {
    gloss: bestGloss,
    confidence: bestConf,
    classId: bestIdx,
    topK: topKResults,
    inferenceTimeMs: inferenceTime,
  };
}
