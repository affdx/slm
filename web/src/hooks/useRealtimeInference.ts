/**
 * Real-time webcam inference hook with optimizations from Python demo.
 *
 * Key optimizations:
 * 1. Continuous frame processing with rolling buffer (deque pattern)
 * 2. Movement detection - only process when user is actively signing
 * 3. Prediction stability filter - require N consistent predictions
 * 4. Hand detection check - only process when hands are visible
 * 5. Higher confidence threshold (0.70)
 */

"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  runInference,
  preloadModel,
  isModelLoaded,
} from "@/lib/inference";
import {
  preloadLandmarkers,
  areLandmarkersReady,
  extractLandmarksFromFrame,
} from "@/lib/landmarks";

// Configuration matching Python demo
const CONFIG = {
  SEQUENCE_LENGTH: 30, // Frames needed for prediction
  CONFIDENCE_THRESHOLD: 0.70, // Minimum confidence to show prediction
  STABLE_COUNT_REQUIRED: 3, // Same prediction needed X times
  MOVEMENT_THRESHOLD: 0.005, // Minimum movement to consider "active"
  PROCESS_INTERVAL_MS: 33, // ~30 FPS processing
  PREDICTION_HISTORY_SIZE: 5, // Size of prediction history buffer
};

export interface RealtimePrediction {
  gloss: string;
  confidence: number;
  isStable: boolean;
  handsDetected: boolean;
  isMoving: boolean;
  bufferProgress: number; // 0-1 progress of frame buffer
}

export interface UseRealtimeInferenceReturn {
  // State
  isReady: boolean;
  isLoading: boolean;
  loadingProgress: string;
  error: string | null;
  isProcessing: boolean;
  prediction: RealtimePrediction | null;

  // Actions
  initialize: () => Promise<void>;
  startProcessing: (videoElement: HTMLVideoElement, canvas: HTMLCanvasElement) => void;
  stopProcessing: () => void;
  reset: () => void;

  // Config
  config: typeof CONFIG;
}

/**
 * Hook for real-time webcam sign language inference
 */
export function useRealtimeInference(): UseRealtimeInferenceReturn {
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [prediction, setPrediction] = useState<RealtimePrediction | null>(null);

  // Refs for processing state (avoid re-renders during processing)
  const processingRef = useRef(false);
  const frameBufferRef = useRef<Float32Array[]>([]);
  const predictionHistoryRef = useRef<Array<{ gloss: string; confidence: number }>>([]);
  const prevLandmarksRef = useRef<Float32Array | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const initializingRef = useRef(false);

  /**
   * Initialize models
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

      setLoadingProgress("Loading ONNX model...");
      await preloadModel();

      setIsReady(true);
      setLoadingProgress("");
      console.log("[RealtimeInference] All models loaded");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load models";
      setError(message);
      console.error("[RealtimeInference] Initialization failed:", err);
    } finally {
      setIsLoading(false);
      initializingRef.current = false;
    }
  }, [isReady]);

  /**
   * Calculate movement between frames
   */
  const calculateMovement = useCallback((current: Float32Array, previous: Float32Array): number => {
    let totalDiff = 0;
    for (let i = 0; i < current.length; i++) {
      totalDiff += Math.abs(current[i] - previous[i]);
    }
    return totalDiff / current.length;
  }, []);

  /**
   * Check if hands are detected in landmarks
   * Hand landmarks are at positions 132-257 (left: 132-194, right: 195-257)
   */
  const checkHandsDetected = useCallback((landmarks: Float32Array): boolean => {
    // Check if either hand has non-zero values
    const POSE_FEATURES = 132;
    const HAND_FEATURES = 63;

    let leftHandSum = 0;
    let rightHandSum = 0;

    for (let i = 0; i < HAND_FEATURES; i++) {
      leftHandSum += Math.abs(landmarks[POSE_FEATURES + i]);
      rightHandSum += Math.abs(landmarks[POSE_FEATURES + HAND_FEATURES + i]);
    }

    // Consider detected if sum is above a small threshold
    return leftHandSum > 0.01 || rightHandSum > 0.01;
  }, []);

  /**
   * Check prediction stability
   */
  const checkStability = useCallback((history: Array<{ gloss: string }>): boolean => {
    if (history.length < CONFIG.STABLE_COUNT_REQUIRED) {
      return false;
    }

    const recent = history.slice(-CONFIG.STABLE_COUNT_REQUIRED);
    const allSame = recent.every(p => p.gloss === recent[0].gloss);
    return allSame;
  }, []);

  /**
   * Process a single frame
   */
  const processFrame = useCallback(async (
    videoElement: HTMLVideoElement,
    canvas: HTMLCanvasElement
  ) => {
    if (!processingRef.current) return;

    try {
      // Draw current frame to canvas
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      
      ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

      // Extract landmarks from current frame
      const landmarks = await extractLandmarksFromFrame(canvas);
      
      // Check if hands are detected
      const handsDetected = checkHandsDetected(landmarks);

      // Calculate movement
      let isMoving = false;
      if (prevLandmarksRef.current) {
        const movement = calculateMovement(landmarks, prevLandmarksRef.current);
        isMoving = movement > CONFIG.MOVEMENT_THRESHOLD;
      }
      prevLandmarksRef.current = landmarks;

      // Only add to buffer if hands detected and moving
      if (handsDetected && isMoving) {
        frameBufferRef.current.push(landmarks);

        // Keep buffer at max size (rolling window)
        if (frameBufferRef.current.length > CONFIG.SEQUENCE_LENGTH) {
          frameBufferRef.current.shift();
        }
      }

      const bufferProgress = frameBufferRef.current.length / CONFIG.SEQUENCE_LENGTH;

      // Run inference if we have enough frames
      if (frameBufferRef.current.length === CONFIG.SEQUENCE_LENGTH) {
        // Flatten the buffer into a single Float32Array
        const flatLandmarks = new Float32Array(CONFIG.SEQUENCE_LENGTH * 258);
        for (let i = 0; i < CONFIG.SEQUENCE_LENGTH; i++) {
          flatLandmarks.set(frameBufferRef.current[i], i * 258);
        }

        // Run inference
        const result = await runInference(flatLandmarks, 5, CONFIG.CONFIDENCE_THRESHOLD);

        // Only consider high-confidence predictions
        if (result.confidence >= CONFIG.CONFIDENCE_THRESHOLD) {
          predictionHistoryRef.current.push({
            gloss: result.gloss,
            confidence: result.confidence,
          });

          // Keep history at max size
          if (predictionHistoryRef.current.length > CONFIG.PREDICTION_HISTORY_SIZE) {
            predictionHistoryRef.current.shift();
          }
        }

        // Check for stable prediction
        const isStable = checkStability(predictionHistoryRef.current);
        const latestPrediction = predictionHistoryRef.current[predictionHistoryRef.current.length - 1];

        if (isStable && latestPrediction) {
          setPrediction({
            gloss: latestPrediction.gloss,
            confidence: latestPrediction.confidence,
            isStable: true,
            handsDetected,
            isMoving,
            bufferProgress,
          });
        } else {
          setPrediction({
            gloss: latestPrediction?.gloss || "...",
            confidence: latestPrediction?.confidence || 0,
            isStable: false,
            handsDetected,
            isMoving,
            bufferProgress,
          });
        }
      } else {
        // Not enough frames yet
        setPrediction({
          gloss: "...",
          confidence: 0,
          isStable: false,
          handsDetected,
          isMoving,
          bufferProgress,
        });
      }

    } catch (err) {
      console.error("[RealtimeInference] Frame processing error:", err);
    }

    // Schedule next frame
    if (processingRef.current) {
      animationFrameRef.current = requestAnimationFrame(() => {
        setTimeout(() => processFrame(videoElement, canvas), CONFIG.PROCESS_INTERVAL_MS);
      });
    }
  }, [checkHandsDetected, calculateMovement, checkStability]);

  /**
   * Start real-time processing
   */
  const startProcessing = useCallback((
    videoElement: HTMLVideoElement,
    canvas: HTMLCanvasElement
  ) => {
    if (!isReady) {
      console.warn("[RealtimeInference] Not ready, initializing first...");
      initialize().then(() => {
        startProcessing(videoElement, canvas);
      });
      return;
    }

    console.log("[RealtimeInference] Starting real-time processing");
    processingRef.current = true;
    setIsProcessing(true);

    // Set canvas size to match video
    canvas.width = videoElement.videoWidth || 640;
    canvas.height = videoElement.videoHeight || 480;

    // Start processing loop
    processFrame(videoElement, canvas);
  }, [isReady, initialize, processFrame]);

  /**
   * Stop processing
   */
  const stopProcessing = useCallback(() => {
    console.log("[RealtimeInference] Stopping processing");
    processingRef.current = false;
    setIsProcessing(false);

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);

  /**
   * Reset state
   */
  const reset = useCallback(() => {
    frameBufferRef.current = [];
    predictionHistoryRef.current = [];
    prevLandmarksRef.current = null;
    setPrediction(null);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopProcessing();
    };
  }, [stopProcessing]);

  // Auto-initialize on mount
  useEffect(() => {
    if (isModelLoaded() && areLandmarkersReady()) {
      setIsReady(true);
      return;
    }
    initialize();
  }, [initialize]);

  return {
    isReady,
    isLoading,
    loadingProgress,
    error,
    isProcessing,
    prediction,
    initialize,
    startProcessing,
    stopProcessing,
    reset,
    config: CONFIG,
  };
}
