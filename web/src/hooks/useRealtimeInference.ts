/**
 * Real-time webcam inference hook with clear state machine and improved UX.
 *
 * State Machine:
 * IDLE → WAITING_FOR_HANDS → COLLECTING → ANALYZING → DETECTED → COOLDOWN → WAITING_FOR_HANDS
 *
 * Key features:
 * 1. Clear state machine with visual feedback
 * 2. Sign history tracking
 * 3. Cooldown period between signs to prevent spam
 * 4. Movement detection for natural signing flow
 * 5. Distance detection (too far/too close) - warning only, doesn't block
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
  | "COLLECTING"        // Collecting frames for inference
  | "ANALYZING"         // Running inference
  | "DETECTED"          // Sign detected! Showing result
  | "COOLDOWN";         // Brief pause before next detection

// Distance status
export type DistanceStatus = "ok" | "too_far" | "too_close" | "unknown";

// Configuration - MUST match model expectations
const CONFIG = {
  SEQUENCE_LENGTH: 30,           // MUST be 30 - model expects 30 frames
  CONFIDENCE_THRESHOLD: 0.70,    // Higher threshold - only accept confident predictions
  STABLE_COUNT_REQUIRED: 1,      // Detect immediately on first high-confidence result
  MOVEMENT_THRESHOLD: 0.002,     // Lower threshold - easier to trigger
  PROCESS_INTERVAL_MS: 33,       // ~30 FPS - collect 30 frames in ~1 second
  PREDICTION_HISTORY_SIZE: 5,    // Size of prediction history buffer
  DETECTION_DISPLAY_MS: 2000,    // How long to show detected sign (2s)
  COOLDOWN_MS: 300,              // Brief cooldown between detections
  MAX_SIGN_HISTORY: 10,          // Maximum signs to keep in history
  MIN_ANALYZING_DISPLAY_MS: 200, // Minimum time to show ANALYZING state
  // Distance thresholds based on shoulder width (normalized 0-1)
  SHOULDER_WIDTH_MIN: 0.12,      // More lenient - too far
  SHOULDER_WIDTH_MAX: 0.50,      // More lenient - too close
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
  isMoving: boolean;
  bufferProgress: number;
  distanceStatus: DistanceStatus;
}

export { type DrawingData } from "@/lib/landmarks";

export interface UseRealtimeInferenceReturn {
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
 * Hook for real-time webcam sign language inference with state machine
 */
export function useRealtimeInference(): UseRealtimeInferenceReturn {
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
  const frameBufferRef = useRef<Float32Array[]>([]);
  const predictionHistoryRef = useRef<Array<{ gloss: string; confidence: number }>>([]);
  const prevLandmarksRef = useRef<Float32Array | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const initializingRef = useRef(false);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const showSkeletonRef = useRef(true);
  const cooldownTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const detectionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const detectionStateRef = useRef<DetectionState>("IDLE");
  const isAnalyzingRef = useRef(false); // Lock to prevent concurrent analysis
  const videoElementRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Helper to update state and ref together (avoids race conditions)
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
   * Check prediction stability
   */
  const checkStability = useCallback((history: Array<{ gloss: string }>): boolean => {
    if (history.length < CONFIG.STABLE_COUNT_REQUIRED) return false;

    const recent = history.slice(-CONFIG.STABLE_COUNT_REQUIRED);
    return recent.every(p => p.gloss === recent[0].gloss);
  }, []);

  /**
   * Transition to detected state and add to history
   */
  const handleSignDetected = useCallback((gloss: string, confidence: number) => {
    const sign: DetectedSign = {
      gloss,
      confidence,
      timestamp: Date.now(),
    };

    setCurrentSign(sign);
    updateDetectionState("DETECTED");
    isAnalyzingRef.current = false;
    
    // Add to history
    setSignHistory(prev => {
      const last = prev[0];
      if (last && last.gloss === gloss && Date.now() - last.timestamp < 3000) {
        return [sign, ...prev.slice(1)].slice(0, CONFIG.MAX_SIGN_HISTORY);
      }
      return [sign, ...prev].slice(0, CONFIG.MAX_SIGN_HISTORY);
    });

    // Clear buffers
    frameBufferRef.current = [];
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
   * Process a single frame
   */
  const processFrame = useCallback(async () => {
    if (!processingRef.current) return;
    
    const videoElement = videoElementRef.current;
    const canvas = canvasRef.current;
    if (!videoElement || !canvas) return;

    const currentState = detectionStateRef.current;

    // During DETECTED or COOLDOWN, just keep drawing skeleton
    if (currentState === "DETECTED" || currentState === "COOLDOWN") {
      try {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
          const { drawingData: currentDrawingData } = await extractLandmarksWithDrawingData(canvas);
          setDrawingData(currentDrawingData);
          
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
        }
      } catch {
        // Ignore
      }
      
      if (processingRef.current) {
        animationFrameRef.current = requestAnimationFrame(() => {
          setTimeout(processFrame, CONFIG.PROCESS_INTERVAL_MS);
        });
      }
      return;
    }

    // If currently analyzing, just draw and wait
    if (isAnalyzingRef.current) {
      try {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
          const { drawingData: currentDrawingData } = await extractLandmarksWithDrawingData(canvas);
          setDrawingData(currentDrawingData);
          
          if (showSkeletonRef.current && overlayCanvasRef.current) {
            const overlayCtx = overlayCanvasRef.current.getContext("2d");
            if (overlayCtx) {
              overlayCtx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
              drawLandmarks(overlayCtx, currentDrawingData, overlayCanvasRef.current.width, overlayCanvasRef.current.height, {
                showPose: true,
                showHands: true,
                poseColor: "rgba(128, 0, 255, 0.7)", // Purple during analyzing
                leftHandColor: "rgba(255, 0, 0, 0.9)",
                rightHandColor: "rgba(0, 100, 255, 0.9)",
                lineWidth: 3,
                pointRadius: 5,
              });
            }
          }
        }
      } catch {
        // Ignore
      }
      
      if (processingRef.current) {
        animationFrameRef.current = requestAnimationFrame(() => {
          setTimeout(processFrame, CONFIG.PROCESS_INTERVAL_MS);
        });
      }
      return;
    }

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

      const handsDetected = checkHandsDetected(landmarks);
      const distanceStatus = checkDistance(landmarks);

      // Calculate movement
      let isMoving = true; // Default to true for first frame
      if (prevLandmarksRef.current) {
        const movement = calculateMovement(landmarks, prevLandmarksRef.current);
        isMoving = movement > CONFIG.MOVEMENT_THRESHOLD;
      }
      prevLandmarksRef.current = landmarks;

      const bufferProgress = frameBufferRef.current.length / CONFIG.SEQUENCE_LENGTH;

      // State machine logic
      if (!handsDetected) {
        updateDetectionState("WAITING_FOR_HANDS");
        frameBufferRef.current = [];
        predictionHistoryRef.current = [];
        setPrediction({
          gloss: "...",
          confidence: 0,
          isStable: false,
          handsDetected: false,
          isMoving: false,
          bufferProgress: 0,
          distanceStatus,
        });
      } else {
        // Hands detected - collect frames (ignore distance for collection, just warn)
        
        // DON'T collect if we're analyzing - wait for analysis to complete
        if (isAnalyzingRef.current) {
          // Just update the prediction display but don't collect
          setPrediction(prev => prev ? { ...prev, handsDetected: true, distanceStatus } : null);
          // Don't proceed with collection
        } else {
          // Safe to collect
          updateDetectionState("COLLECTING");
          frameBufferRef.current.push(landmarks);

          if (frameBufferRef.current.length > CONFIG.SEQUENCE_LENGTH) {
            frameBufferRef.current.shift();
          }

          const bufferProgress = frameBufferRef.current.length / CONFIG.SEQUENCE_LENGTH;
          
          // Only log every 10 frames to reduce noise
          if (frameBufferRef.current.length % 10 === 0 || frameBufferRef.current.length === CONFIG.SEQUENCE_LENGTH) {
            console.log("[RealtimeInference] Buffer:", frameBufferRef.current.length, "/", CONFIG.SEQUENCE_LENGTH);
          }
          
          setPrediction({
            gloss: "...",
            confidence: 0,
            isStable: false,
            handsDetected: true,
            isMoving,
            bufferProgress,
            distanceStatus,
          });

          // Run inference if we have enough frames
          if (frameBufferRef.current.length >= CONFIG.SEQUENCE_LENGTH) {
            // Lock to prevent re-entry
            isAnalyzingRef.current = true;
            updateDetectionState("ANALYZING");
            const inferenceStartTime = Date.now();
            console.log("[RealtimeInference] Starting inference...");

            // Copy buffer for inference (so we can clear immediately)
            const framesToAnalyze = [...frameBufferRef.current];
            
            // Clear buffer immediately to restart collection
            frameBufferRef.current = [];

            const flatLandmarks = new Float32Array(CONFIG.SEQUENCE_LENGTH * 258);
            for (let i = 0; i < CONFIG.SEQUENCE_LENGTH; i++) {
              flatLandmarks.set(framesToAnalyze[i], i * 258);
            }

            // Run inference
            runInference(flatLandmarks, 5, CONFIG.CONFIDENCE_THRESHOLD)
              .then(async (result) => {
                console.log("[RealtimeInference] Result:", result.gloss, (result.confidence * 100).toFixed(0) + "%");

                // Ensure minimum display time for ANALYZING state
                const elapsed = Date.now() - inferenceStartTime;
                if (elapsed < CONFIG.MIN_ANALYZING_DISPLAY_MS) {
                  await new Promise(r => setTimeout(r, CONFIG.MIN_ANALYZING_DISPLAY_MS - elapsed));
                }

                if (!processingRef.current) {
                  isAnalyzingRef.current = false;
                  return;
                }

                // Always show the result to user (even low confidence)
                setPrediction({
                  gloss: result.gloss,
                  confidence: result.confidence,
                  isStable: false,
                  handsDetected: true,
                  isMoving: true,
                  bufferProgress: 0,
                  distanceStatus: "ok",
                });

                if (result.confidence >= CONFIG.CONFIDENCE_THRESHOLD) {
                  predictionHistoryRef.current.push({
                    gloss: result.gloss,
                    confidence: result.confidence,
                  });

                  if (predictionHistoryRef.current.length > CONFIG.PREDICTION_HISTORY_SIZE) {
                    predictionHistoryRef.current.shift();
                  }

                  const isStable = checkStability(predictionHistoryRef.current);
                  
                  if (isStable) {
                    console.log("[RealtimeInference] DETECTED:", result.gloss);
                    handleSignDetected(result.gloss, result.confidence);
                    return; // handleSignDetected will manage state
                  }
                } else {
                  // Low confidence - clear prediction history
                  predictionHistoryRef.current = [];
                }

                // Resume collecting
                isAnalyzingRef.current = false;
                updateDetectionState("COLLECTING");
              })
              .catch((err) => {
                console.error("[RealtimeInference] Inference error:", err);
                isAnalyzingRef.current = false;
                updateDetectionState("COLLECTING");
              });
          }
        }
      }

    } catch (err) {
      console.error("[RealtimeInference] Frame processing error:", err);
    }

    // Schedule next frame
    if (processingRef.current) {
      animationFrameRef.current = requestAnimationFrame(() => {
        setTimeout(processFrame, CONFIG.PROCESS_INTERVAL_MS);
      });
    }
  }, [checkHandsDetected, checkDistance, calculateMovement, checkStability, handleSignDetected, updateDetectionState]);

  /**
   * Start real-time processing
   */
  const startProcessing = useCallback((
    videoElement: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    overlayCanvas?: HTMLCanvasElement
  ) => {
    if (!isReady) {
      console.warn("[RealtimeInference] Not ready, initializing first...");
      initialize().then(() => {
        startProcessing(videoElement, canvas, overlayCanvas);
      });
      return;
    }

    console.log("[RealtimeInference] Starting real-time processing");
    processingRef.current = true;
    isAnalyzingRef.current = false;
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
    console.log("[RealtimeInference] Stopping processing");
    processingRef.current = false;
    isAnalyzingRef.current = false;
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
    prevLandmarksRef.current = null;
    isAnalyzingRef.current = false;
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
