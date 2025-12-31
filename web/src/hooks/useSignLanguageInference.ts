/**
 * React hook for client-side sign language inference.
 *
 * Combines MediaPipe landmark extraction with ONNX Runtime inference
 * for fully client-side sign language translation.
 */

"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  runInference,
  preloadModel,
  isModelLoaded,
  getGlosses,
  getCurrentModelType,
  setModelType,
  getAvailableModels,
  type ModelType,
  type ModelConfig,
} from "@/lib/inference";
import {
  extractLandmarksFromBlob,
  preloadLandmarkers,
  areLandmarkersReady,
  getCurrentDelegate,
  setDelegate,
} from "@/lib/landmarks";

export interface PredictionResult {
  predicted_gloss: string;
  confidence: number;
  top_5_predictions: Array<{
    gloss: string;
    confidence: number;
  }>;
  processing_time_ms: number;
  landmark_extraction_time_ms: number;
  inference_time_ms: number;
}

export interface UseSignLanguageInferenceReturn {
  // State
  isReady: boolean;
  isLoading: boolean;
  loadingProgress: string;
  error: string | null;
  delegate: "GPU" | "CPU";
  currentModel: ModelType;
  availableModels: Array<{ type: ModelType; config: ModelConfig }>;

  // Actions
  predict: (videoBlob: Blob) => Promise<PredictionResult>;
  initialize: () => Promise<void>;
  switchDelegate: (delegate: "GPU" | "CPU") => Promise<void>;
  switchModel: (modelType: ModelType) => Promise<void>;

  // Utilities
  getGlossList: () => Promise<string[]>;
}

/**
 * Hook for managing sign language inference
 */
export function useSignLanguageInference(): UseSignLanguageInferenceReturn {
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [delegate, setDelegateState] = useState<"GPU" | "CPU">("GPU");
  const [currentModel, setCurrentModel] = useState<ModelType>(getCurrentModelType());
  const initializingRef = useRef(false);
  
  const availableModels = getAvailableModels();

  /**
   * Initialize models (ONNX + MediaPipe)
   */
  const initialize = useCallback(async () => {
    if (initializingRef.current || isReady) {
      return;
    }

    initializingRef.current = true;
    setIsLoading(true);
    setError(null);

    try {
      setLoadingProgress("Loading MediaPipe models...");
      await preloadLandmarkers();
      setDelegateState(getCurrentDelegate());

      setLoadingProgress("Loading ONNX model...");
      await preloadModel();

      setIsReady(true);
      setLoadingProgress("");
      console.log("[Hook] All models loaded and ready");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load models";
      setError(message);
      console.error("[Hook] Initialization failed:", err);
    } finally {
      setIsLoading(false);
      initializingRef.current = false;
    }
  }, [isReady]);

  /**
   * Run prediction on a video blob
   */
  const predict = useCallback(
    async (videoBlob: Blob): Promise<PredictionResult> => {
      // Initialize if not ready
      if (!isReady) {
        await initialize();
      }

      const totalStartTime = performance.now();

      // Extract landmarks
      console.log("[Hook] Starting landmark extraction...");
      const landmarkStartTime = performance.now();
      const landmarks = await extractLandmarksFromBlob(videoBlob);
      const landmarkTime = performance.now() - landmarkStartTime;

      // Run inference
      console.log("[Hook] Running inference...");
      const result = await runInference(landmarks, 5, 0.3);

      const totalTime = performance.now() - totalStartTime;

      // Format result to match API format
      return {
        predicted_gloss: result.gloss,
        confidence: result.confidence,
        top_5_predictions: result.topK.map((item) => ({
          gloss: item.gloss,
          confidence: item.confidence,
        })),
        processing_time_ms: totalTime,
        landmark_extraction_time_ms: landmarkTime,
        inference_time_ms: result.inferenceTimeMs,
      };
    },
    [isReady, initialize]
  );

  /**
   * Get list of all glosses
   */
  const getGlossList = useCallback(async (): Promise<string[]> => {
    return getGlosses();
  }, []);

  /**
   * Switch between GPU and CPU delegate for MediaPipe
   */
  const switchDelegate = useCallback(async (newDelegate: "GPU" | "CPU") => {
    if (newDelegate === delegate) return;
    
    setIsLoading(true);
    setLoadingProgress(`Switching to ${newDelegate} processing...`);
    setError(null);

    try {
      await setDelegate(newDelegate);
      setDelegateState(newDelegate);
      setLoadingProgress("");
      console.log(`[Hook] Switched to ${newDelegate} delegate`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to switch delegate";
      setError(message);
      console.error("[Hook] Failed to switch delegate:", err);
    } finally {
      setIsLoading(false);
    }
  }, [delegate]);

  /**
   * Switch between models
   */
  const switchModel = useCallback(async (modelType: ModelType) => {
    if (modelType === currentModel) return;
    
    setIsLoading(true);
    const modelConfig = availableModels.find(m => m.type === modelType)?.config;
    setLoadingProgress(`Loading ${modelConfig?.name || modelType}...`);
    setError(null);

    try {
      await setModelType(modelType);
      setCurrentModel(modelType);
      setLoadingProgress("");
      console.log(`[Hook] Switched to ${modelType} model`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to switch model";
      setError(message);
      console.error("[Hook] Failed to switch model:", err);
    } finally {
      setIsLoading(false);
    }
  }, [currentModel, availableModels]);

  // Auto-initialize on mount (optional - can be disabled for lazy loading)
  useEffect(() => {
    // Check if already loaded
    if (isModelLoaded() && areLandmarkersReady()) {
      setIsReady(true);
      return;
    }

    // Start loading in background
    initialize();
  }, [initialize]);

  return {
    isReady,
    isLoading,
    loadingProgress,
    error,
    delegate,
    currentModel,
    availableModels,
    predict,
    initialize,
    switchDelegate,
    switchModel,
    getGlossList,
  };
}

// Re-export formatting utilities for backward compatibility
export { formatGlossName, formatConfidence } from "@/lib/format";
