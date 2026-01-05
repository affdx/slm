/**
 * Real-time webcam inference hook V2 - Python-style sliding window.
 *
 * Key differences from V1:
 * 1. Sliding window with overlap (like Python's deque(maxlen=30))
 *    - Buffer maintains temporal continuity, shifts by 1 frame
 *    - Overlaps 29/30 frames between predictions
 * 2. Predict every N frames (configurable, default 3)
 *    - Much higher prediction frequency (~10/sec vs ~1/sec)
 * 3. Non-blocking inference
 *    - Frame collection continues during inference
 *
 * State Machine (simplified from V1):
 * IDLE -> WAITING_FOR_HANDS -> COLLECTING -> (continuous predictions) -> DETECTED -> COOLDOWN
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
  extractLandmarksWithDrawingData,
  drawLandmarks,
  type DrawingData,
} from "@/lib/landmarks";

// Detection states
export type DetectionState = 
  | "IDLE"              // Not started
  | "WAITING_FOR_HANDS" // Looking for hands in frame
  | "COLLECTING"        // Collecting frames, running inference
  | "DETECTED"          // Sign detected! Showing result
  | "COOLDOWN";         // Brief pause before next detection

// Distance status
export type DistanceStatus = "ok" | "too_far" | "too_close" | "unknown";

// Configuration - Python-style sliding window
const CONFIG = {
  SEQUENCE_LENGTH: 30,           // MUST be 30 - model expects 30 frames
  CONFIDENCE_THRESHOLD: 0.70,    // High confidence threshold
  PRED_EVERY_N_FRAMES: 3,        // Predict every N frames (Python uses 2)
  PROCESS_INTERVAL_MS: 33,       // ~30 FPS frame capture
  DETECTION_DISPLAY_MS: 1500,    // How long to show detected sign
  COOLDOWN_MS: 200,              // Brief cooldown between detections
  MAX_SIGN_HISTORY: 10,          // Maximum signs to keep in history
  // Distance thresholds based on shoulder width (normalized 0-1)
  SHOULDER_WIDTH_MIN: 0.12,      // Too far
  SHOULDER_WIDTH_MAX: 0.50,      // Too close
  // Stability: require N consecutive same predictions
  STABLE_COUNT_REQUIRED: 2,      // Require 2 consecutive same predictions
  PREDICTION_HISTORY_SIZE: 5,    // Keep last N predictions for stability check
};

export interface DetectedSign {
  gloss: string;
  confidence: number;
  timestamp: number;
}

export interface RealtimePrediction {
  gloss: string;
  confidence: number;
  isStable: boolean;
  handsDetected: boolean;
  bufferProgress: number;
  distanceStatus: DistanceStatus;
  frameCount: number;
  predictionsPerSecond: number;
}

export { type DrawingData } from "@/lib/landmarks";

export interface UseRealtimeInferenceV2Return {
  // State
  isReady: boolean;
  isLoading: boolean;
  loadingProgress: string;
  error: string | null;
  isProcessing: boolean;
  detectionState: DetectionState;
  prediction: RealtimePrediction | null;
  currentSign: DetectedSign | null;
  signHistory: DetectedSign[];
  drawingData: DrawingData | null;

  // Actions
  initialize: () => Promise<void>;
  startProcessing: (videoElement: HTMLVideoElement, canvas: HTMLCanvasElement, overlayCanvas?: HTMLCanvasElement) => void;
  stopProcessing: () => void;
  reset: () => void;
  clearHistory: () => void;
  setShowSkeleton: (show: boolean) => void;

  // Config
  config: typeof CONFIG;
  showSkeleton: boolean;
}

/**
 * Hook for real-time webcam sign language inference with Python-style sliding window
 */
export function useRealtimeInferenceV2(): UseRealtimeInferenceV2Return {
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectionState, setDetectionState] = useState<DetectionState>("IDLE");
  const [prediction, setPrediction] = useState<RealtimePrediction | null>(null);
  const [currentSign, setCurrentSign] = useState<DetectedSign | null>(null);
  const [signHistory, setSignHistory] = useState<DetectedSign[]>([]);
  const [drawingData, setDrawingData] = useState<DrawingData | null>(null);
  const [showSkeleton, setShowSkeleton] = useState(true);

  // Refs for processing state
  const processingRef = useRef(false);
  const frameBufferRef = useRef<Float32Array[]>([]); // Sliding window buffer
  const predictionHistoryRef = useRef<Array<{ gloss: string; confidence: number }>>([]);
  const animationFrameRef = useRef<number | null>(null);
  const initializingRef = useRef(false);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const showSkeletonRef = useRef(true);
  const cooldownTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const detectionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const detectionStateRef = useRef<DetectionState>("IDLE");
  const isInferenceRunningRef = useRef(false); // Track if inference is in progress
  const videoElementRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameCountRef = useRef(0);
  const predictionCountRef = useRef(0);
  const lastPredictionTimeRef = useRef(0);
  const predictionsPerSecondRef = useRef(0);

  // Helper to update state and ref together
  const updateDetectionState = useCallback((newState: DetectionState) => {
    detectionStateRef.current = newState;
    setDetectionState(newState);
  }, []);

  /**
   * Initialize models
   */
  const initialize = useCallback(async () => {
    if (initializingRef.current || isReady) return;

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
      console.log("[RealtimeInferenceV2] All models loaded");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load models";
      setError(message);
      console.error("[RealtimeInferenceV2] Initialization failed:", err);
    } finally {
      setIsLoading(false);
      initializingRef.current = false;
    }
  }, [isReady]);

  /**
   * Check if hands are detected in landmarks
   */
  const checkHandsDetected = useCallback((landmarks: Float32Array): boolean => {
    const POSE_FEATURES = 132;
    const HAND_FEATURES = 63;

    let leftHandSum = 0;
    let rightHandSum = 0;

    for (let i = 0; i < HAND_FEATURES; i++) {
      leftHandSum += Math.abs(landmarks[POSE_FEATURES + i]);
      rightHandSum += Math.abs(landmarks[POSE_FEATURES + HAND_FEATURES + i]);
    }

    return leftHandSum > 0.01 || rightHandSum > 0.01;
  }, []);

  /**
   * Check distance from camera based on shoulder width
   */
  const checkDistance = useCallback((landmarks: Float32Array): DistanceStatus => {
    const LEFT_SHOULDER_IDX = 11 * 4;
    const RIGHT_SHOULDER_IDX = 12 * 4;

    const leftShoulderX = landmarks[LEFT_SHOULDER_IDX];
    const leftShoulderY = landmarks[LEFT_SHOULDER_IDX + 1];
    const leftShoulderVis = landmarks[LEFT_SHOULDER_IDX + 3];

    const rightShoulderX = landmarks[RIGHT_SHOULDER_IDX];
    const rightShoulderY = landmarks[RIGHT_SHOULDER_IDX + 1];
    const rightShoulderVis = landmarks[RIGHT_SHOULDER_IDX + 3];

    if (leftShoulderVis < 0.5 || rightShoulderVis < 0.5) {
      return "unknown";
    }

    const shoulderWidth = Math.sqrt(
      Math.pow(rightShoulderX - leftShoulderX, 2) +
      Math.pow(rightShoulderY - leftShoulderY, 2)
    );

    if (shoulderWidth < CONFIG.SHOULDER_WIDTH_MIN) {
      return "too_far";
    } else if (shoulderWidth > CONFIG.SHOULDER_WIDTH_MAX) {
      return "too_close";
    }
    return "ok";
  }, []);

  /**
   * Check prediction stability (consecutive same predictions)
   */
  const checkStability = useCallback((history: Array<{ gloss: string }>): boolean => {
    if (history.length < CONFIG.STABLE_COUNT_REQUIRED) return false;

    const recent = history.slice(-CONFIG.STABLE_COUNT_REQUIRED);
    return recent.every(p => p.gloss === recent[0].gloss);
  }, []);

  /**
   * Handle sign detection
   */
  const handleSignDetected = useCallback((gloss: string, confidence: number) => {
    const sign: DetectedSign = {
      gloss,
      confidence,
      timestamp: Date.now(),
    };

    setCurrentSign(sign);
    updateDetectionState("DETECTED");
    
    // Add to history (dedupe if same sign within 3 seconds)
    setSignHistory(prev => {
      const last = prev[0];
      if (last && last.gloss === gloss && Date.now() - last.timestamp < 3000) {
        return [sign, ...prev.slice(1)].slice(0, CONFIG.MAX_SIGN_HISTORY);
      }
      return [sign, ...prev].slice(0, CONFIG.MAX_SIGN_HISTORY);
    });

    // Clear prediction history for next sign
    predictionHistoryRef.current = [];

    // Show detection, then cooldown
    if (detectionTimeoutRef.current) clearTimeout(detectionTimeoutRef.current);
    detectionTimeoutRef.current = setTimeout(() => {
      updateDetectionState("COOLDOWN");
      setCurrentSign(null);
      
      if (cooldownTimeoutRef.current) clearTimeout(cooldownTimeoutRef.current);
      cooldownTimeoutRef.current = setTimeout(() => {
        if (processingRef.current) {
          updateDetectionState("WAITING_FOR_HANDS");
        }
      }, CONFIG.COOLDOWN_MS);
    }, CONFIG.DETECTION_DISPLAY_MS);
  }, [updateDetectionState]);

  /**
   * Run inference on current buffer (non-blocking)
   */
  const runInferenceOnBuffer = useCallback(async () => {
    if (isInferenceRunningRef.current) return;
    if (frameBufferRef.current.length < CONFIG.SEQUENCE_LENGTH) return;

    isInferenceRunningRef.current = true;

    try {
      // Copy buffer for inference (don't clear - sliding window!)
      const framesToAnalyze = frameBufferRef.current.slice(-CONFIG.SEQUENCE_LENGTH);

      const flatLandmarks = new Float32Array(CONFIG.SEQUENCE_LENGTH * 258);
      for (let i = 0; i < CONFIG.SEQUENCE_LENGTH; i++) {
        flatLandmarks.set(framesToAnalyze[i], i * 258);
      }

      const result = await runInference(flatLandmarks, 5, CONFIG.CONFIDENCE_THRESHOLD);
      
      // Update predictions per second
      predictionCountRef.current++;
      const now = Date.now();
      if (now - lastPredictionTimeRef.current >= 1000) {
        predictionsPerSecondRef.current = predictionCountRef.current;
        predictionCountRef.current = 0;
        lastPredictionTimeRef.current = now;
      }

      // Only process if still running and not in detection/cooldown
      if (!processingRef.current) return;
      if (detectionStateRef.current === "DETECTED" || detectionStateRef.current === "COOLDOWN") return;

      // Log prediction
      console.log(`[RealtimeInferenceV2] frame=${frameCountRef.current} Pred: ${result.gloss} | prob=${(result.confidence * 100).toFixed(1)}%`);

      // Update prediction display
      setPrediction(prev => prev ? {
        ...prev,
        gloss: result.gloss,
        confidence: result.confidence,
        predictionsPerSecond: predictionsPerSecondRef.current,
      } : null);

      // Check for high-confidence stable prediction
      if (result.confidence >= CONFIG.CONFIDENCE_THRESHOLD && result.gloss !== "unknown") {
        predictionHistoryRef.current.push({
          gloss: result.gloss,
          confidence: result.confidence,
        });

        if (predictionHistoryRef.current.length > CONFIG.PREDICTION_HISTORY_SIZE) {
          predictionHistoryRef.current.shift();
        }

        const isStable = checkStability(predictionHistoryRef.current);
        
        if (isStable) {
          console.log(`[RealtimeInferenceV2] DETECTED: ${result.gloss}`);
          handleSignDetected(result.gloss, result.confidence);
        }
      } else {
        // Low confidence - keep history but don't add
      }

    } catch (err) {
      console.error("[RealtimeInferenceV2] Inference error:", err);
    } finally {
      isInferenceRunningRef.current = false;
    }
  }, [checkStability, handleSignDetected]);

  /**
   * Process a single frame - Python-style sliding window
   */
  const processFrame = useCallback(async () => {
    if (!processingRef.current) return;
    
    const videoElement = videoElementRef.current;
    const canvas = canvasRef.current;
    if (!videoElement || !canvas) return;

    const currentState = detectionStateRef.current;
    frameCountRef.current++;

    try {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      
      ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

      const { features: landmarks, drawingData: currentDrawingData } = await extractLandmarksWithDrawingData(canvas);
      
      setDrawingData(currentDrawingData);

      // Draw skeleton overlay
      if (showSkeletonRef.current && overlayCanvasRef.current) {
        const overlayCtx = overlayCanvasRef.current.getContext("2d");
        if (overlayCtx) {
          overlayCtx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
          drawLandmarks(overlayCtx, currentDrawingData, overlayCanvasRef.current.width, overlayCanvasRef.current.height, {
            showPose: true,
            showHands: true,
            poseColor: "rgba(0, 255, 0, 0.7)",
            leftHandColor: "rgba(255, 0, 0, 0.9)",
            rightHandColor: "rgba(0, 100, 255, 0.9)",
            lineWidth: 3,
            pointRadius: 5,
          });
        }
      }

      // During DETECTED or COOLDOWN, just draw skeleton, don't collect
      if (currentState === "DETECTED" || currentState === "COOLDOWN") {
        if (processingRef.current) {
          animationFrameRef.current = requestAnimationFrame(() => {
            setTimeout(processFrame, CONFIG.PROCESS_INTERVAL_MS);
          });
        }
        return;
      }

      const handsDetected = checkHandsDetected(landmarks);
      const distanceStatus = checkDistance(landmarks);
      const bufferProgress = Math.min(frameBufferRef.current.length / CONFIG.SEQUENCE_LENGTH, 1);

      if (!handsDetected) {
        // No hands - reset to waiting state
        updateDetectionState("WAITING_FOR_HANDS");
        frameBufferRef.current = []; // Clear buffer when hands disappear
        predictionHistoryRef.current = [];
        setPrediction({
          gloss: "...",
          confidence: 0,
          isStable: false,
          handsDetected: false,
          bufferProgress: 0,
          distanceStatus,
          frameCount: frameCountRef.current,
          predictionsPerSecond: predictionsPerSecondRef.current,
        });
      } else {
        // Hands detected - SLIDING WINDOW: always add frame
        updateDetectionState("COLLECTING");
        
        // Add frame to buffer (sliding window - keep last SEQUENCE_LENGTH frames)
        frameBufferRef.current.push(landmarks);
        if (frameBufferRef.current.length > CONFIG.SEQUENCE_LENGTH) {
          frameBufferRef.current.shift(); // Remove oldest, keep 30
        }

        const newBufferProgress = frameBufferRef.current.length / CONFIG.SEQUENCE_LENGTH;
        
        setPrediction({
          gloss: prediction?.gloss || "...",
          confidence: prediction?.confidence || 0,
          isStable: false,
          handsDetected: true,
          bufferProgress: newBufferProgress,
          distanceStatus,
          frameCount: frameCountRef.current,
          predictionsPerSecond: predictionsPerSecondRef.current,
        });

        // Run inference every N frames when buffer is full (Python-style)
        if (frameBufferRef.current.length >= CONFIG.SEQUENCE_LENGTH && 
            frameCountRef.current % CONFIG.PRED_EVERY_N_FRAMES === 0) {
          // Non-blocking inference
          runInferenceOnBuffer();
        }
      }

    } catch (err) {
      console.error("[RealtimeInferenceV2] Frame processing error:", err);
    }

    // Schedule next frame
    if (processingRef.current) {
      animationFrameRef.current = requestAnimationFrame(() => {
        setTimeout(processFrame, CONFIG.PROCESS_INTERVAL_MS);
      });
    }
  }, [checkHandsDetected, checkDistance, updateDetectionState, runInferenceOnBuffer, prediction]);

  /**
   * Start real-time processing
   */
  const startProcessing = useCallback((
    videoElement: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    overlayCanvas?: HTMLCanvasElement
  ) => {
    if (!isReady) {
      console.warn("[RealtimeInferenceV2] Not ready, initializing first...");
      initialize().then(() => {
        startProcessing(videoElement, canvas, overlayCanvas);
      });
      return;
    }

    console.log("[RealtimeInferenceV2] Starting real-time processing (Python-style sliding window)");
    processingRef.current = true;
    isInferenceRunningRef.current = false;
    frameCountRef.current = 0;
    predictionCountRef.current = 0;
    lastPredictionTimeRef.current = Date.now();
    predictionsPerSecondRef.current = 0;
    setIsProcessing(true);
    updateDetectionState("WAITING_FOR_HANDS");

    // Store refs
    videoElementRef.current = videoElement;
    canvasRef.current = canvas;

    canvas.width = videoElement.videoWidth || 640;
    canvas.height = videoElement.videoHeight || 480;

    if (overlayCanvas) {
      overlayCanvas.width = videoElement.videoWidth || 640;
      overlayCanvas.height = videoElement.videoHeight || 480;
      overlayCanvasRef.current = overlayCanvas;
    }

    processFrame();
  }, [isReady, initialize, processFrame, updateDetectionState]);

  /**
   * Stop processing
   */
  const stopProcessing = useCallback(() => {
    console.log("[RealtimeInferenceV2] Stopping processing");
    processingRef.current = false;
    isInferenceRunningRef.current = false;
    setIsProcessing(false);
    updateDetectionState("IDLE");

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (cooldownTimeoutRef.current) {
      clearTimeout(cooldownTimeoutRef.current);
      cooldownTimeoutRef.current = null;
    }
    if (detectionTimeoutRef.current) {
      clearTimeout(detectionTimeoutRef.current);
      detectionTimeoutRef.current = null;
    }
    
    // Clear overlay
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
      }
    }
  }, [updateDetectionState]);

  /**
   * Reset state
   */
  const reset = useCallback(() => {
    frameBufferRef.current = [];
    predictionHistoryRef.current = [];
    isInferenceRunningRef.current = false;
    frameCountRef.current = 0;
    predictionCountRef.current = 0;
    predictionsPerSecondRef.current = 0;
    setPrediction(null);
    setCurrentSign(null);
    setDrawingData(null);
    updateDetectionState("IDLE");
    
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
      }
    }
  }, [updateDetectionState]);

  /**
   * Clear sign history
   */
  const clearHistory = useCallback(() => {
    setSignHistory([]);
  }, []);

  /**
   * Toggle skeleton visibility
   */
  const handleSetShowSkeleton = useCallback((show: boolean) => {
    setShowSkeleton(show);
    showSkeletonRef.current = show;
    
    if (!show && overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
      }
    }
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
    detectionState,
    prediction,
    currentSign,
    signHistory,
    drawingData,
    initialize,
    startProcessing,
    stopProcessing,
    reset,
    clearHistory,
    setShowSkeleton: handleSetShowSkeleton,
    config: CONFIG,
    showSkeleton,
  };
}
