/**
 * ONNX Runtime Web inference module for sign language classification.
 *
 * Supports multiple models with per-feature normalization.
 * Uses WebGL backend for GPU acceleration in the browser.
 * Uses dynamic imports to avoid SSR issues.
 */

// Model configuration
export type ModelType = "baseline" | "improved";

export interface ModelConfig {
  name: string;
  description: string;
  modelPath: string;
  normStatsPath: string;
}

export const MODEL_CONFIGS: Record<ModelType, ModelConfig> = {
  baseline: {
    name: "Baseline BiLSTM",
    description: "Original BiLSTM with attention (92.75% accuracy)",
    modelPath: "/model_baseline.onnx",
    normStatsPath: "/model_baseline_norm_stats.json",
  },
  improved: {
    name: "Improved BiLSTM",
    description: "Optimized training pipeline (92.75% accuracy)",
    modelPath: "/model_npy.onnx",
    normStatsPath: "/model_npy_norm_stats.json",
  },
};

const CLASS_MAPPING_PATH = "/class_mapping.json";
const NUM_FRAMES = 30;
const NUM_FEATURES = 258;

// Current selected model
let currentModelType: ModelType = "baseline";

// ONNX Runtime types (imported dynamically)
interface OrtSession {
  inputNames: string[];
  outputNames: string[];
  run(
    feeds: Record<string, unknown>
  ): Promise<Record<string, { data: Float32Array }>>;
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

interface NormStats {
  mean: number[];
  std: number[];
  shape: number[];
}

// Singleton instances per model type
const sessions: Map<ModelType, Promise<OrtSession>> = new Map();
const normStatsCache: Map<ModelType, NormStats> = new Map();
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

  const cdnUrl =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js";

  // Check if already loaded (from script tag)
  if (
    typeof window !== "undefined" &&
    (window as unknown as { ort: OrtModule }).ort
  ) {
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
    script.onerror = () =>
      reject(new Error("Failed to load ONNX Runtime from CDN"));
    document.head.appendChild(script);
  });

  // Wait a bit for the script to initialize
  await new Promise((resolve) => setTimeout(resolve, 100));

  if (
    typeof window !== "undefined" &&
    (window as unknown as { ort: OrtModule }).ort
  ) {
    ortModule = (window as unknown as { ort: OrtModule }).ort;

    // Configure ONNX Runtime - use WASM backend for reliability
    ortModule.env.wasm.numThreads = 1;
    ortModule.env.wasm.simd = true;

    console.log("[ONNX] ONNX Runtime loaded from CDN");
    return ortModule;
  }

  throw new Error("ONNX Runtime not available after loading script");
}

/**
 * Load normalization stats from JSON for a specific model
 */
async function loadNormStats(modelType: ModelType): Promise<NormStats> {
  const cached = normStatsCache.get(modelType);
  if (cached) {
    return cached;
  }

  const config = MODEL_CONFIGS[modelType];
  console.log(`[ONNX] Loading norm stats for ${config.name}...`);
  const response = await fetch(config.normStatsPath);
  if (!response.ok) {
    throw new Error(`Failed to load norm stats: ${response.statusText}`);
  }

  const stats = await response.json();
  normStatsCache.set(modelType, stats);
  console.log(`[ONNX] Loaded norm stats: ${stats.mean.length} features`);
  return stats;
}

/**
 * Load the ONNX model session for a specific model type (singleton pattern per model)
 */
export async function loadModel(modelType?: ModelType): Promise<OrtSession> {
  const type = modelType ?? currentModelType;
  
  const existingSession = sessions.get(type);
  if (existingSession) {
    return existingSession;
  }

  const config = MODEL_CONFIGS[type];
  
  const sessionPromise = (async () => {
    const ort = await getOrt();

    console.log(`[ONNX] Loading ${config.name} from:`, config.modelPath);
    const startTime = performance.now();

    try {
      // Use WASM backend for reliability (WebGL can have issues)
      const session = await ort.InferenceSession.create(config.modelPath, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });

      const loadTime = performance.now() - startTime;
      console.log(`[ONNX] ${config.name} loaded in ${loadTime.toFixed(0)}ms`);
      console.log("[ONNX] Input names:", session.inputNames);
      console.log("[ONNX] Output names:", session.outputNames);

      return session;
    } catch (error) {
      console.error(`[ONNX] Failed to load ${config.name}:`, error);
      sessions.delete(type);
      throw error;
    }
  })();

  sessions.set(type, sessionPromise);
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
 * Normalize landmarks using per-feature normalization
 */
function normalizeLandmarks(
  landmarks: Float32Array,
  stats: NormStats
): Float32Array {
  const normalized = new Float32Array(landmarks.length);
  const numFeatures = stats.mean.length;

  // landmarks is [NUM_FRAMES * NUM_FEATURES] flattened
  // stats.mean and stats.std are [NUM_FEATURES]
  for (let i = 0; i < landmarks.length; i++) {
    const featureIdx = i % numFeatures;
    const mean = stats.mean[featureIdx];
    const std = stats.std[featureIdx];
    normalized[i] = (landmarks[i] - mean) / std;
  }

  return normalized;
}

/**
 * Softmax function
 */
function softmax(logits: Float32Array): Float32Array {
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
 * @param modelType - Which model to use (defaults to current model)
 */
export async function runInference(
  landmarks: Float32Array,
  topK: number = 5,
  confidenceThreshold: number = 0.3,
  modelType?: ModelType
): Promise<InferenceResult> {
  const type = modelType ?? currentModelType;
  
  // Validate input
  const expectedLength = NUM_FRAMES * NUM_FEATURES;
  if (landmarks.length !== expectedLength) {
    throw new Error(
      `Invalid landmarks length: expected ${expectedLength}, got ${landmarks.length}`
    );
  }

  // Load model, norm stats, and class mapping if needed
  const [session, stats, ort] = await Promise.all([
    loadModel(type),
    loadNormStats(type),
    loadClassMapping().then(() => getOrt()),
  ]);

  // Normalize landmarks using per-feature normalization
  const normalizedLandmarks = normalizeLandmarks(landmarks, stats);

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
    `[ONNX] Inference in ${inferenceTime.toFixed(1)}ms - ${bestGloss} (${(bestConf * 100).toFixed(1)}%)`
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
 * Check if a model is loaded
 */
export function isModelLoaded(modelType?: ModelType): boolean {
  const type = modelType ?? currentModelType;
  return sessions.has(type);
}

/**
 * Get the current model type
 */
export function getCurrentModelType(): ModelType {
  return currentModelType;
}

/**
 * Set the current model type
 */
export async function setModelType(modelType: ModelType): Promise<void> {
  currentModelType = modelType;
  // Preload the new model
  await loadModel(modelType);
  console.log(`[ONNX] Switched to ${MODEL_CONFIGS[modelType].name}`);
}

/**
 * Get available model types and their configs
 */
export function getAvailableModels(): Array<{ type: ModelType; config: ModelConfig }> {
  return Object.entries(MODEL_CONFIGS).map(([type, config]) => ({
    type: type as ModelType,
    config,
  }));
}

/**
 * Get list of all glosses
 */
export async function getGlosses(): Promise<string[]> {
  const mapping = await loadClassMapping();
  return Object.keys(mapping).sort();
}

/**
 * Preload a model (call on app init for faster first inference)
 */
export async function preloadModel(modelType?: ModelType): Promise<void> {
  const type = modelType ?? currentModelType;
  await Promise.all([loadModel(type), loadNormStats(type), loadClassMapping()]);
  console.log(`[ONNX] ${MODEL_CONFIGS[type].name}, norm stats, and class mapping preloaded`);
}
