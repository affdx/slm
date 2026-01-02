/**
 * Video player with real-time inference overlay.
 * 
 * Similar to baseline/Model_Test.py - processes video frame-by-frame
 * with live prediction overlay, top-k results, buffer progress, and FPS.
 * 
 * Key behavior (matching Python baseline):
 * - Processes frames sequentially (not at video's native FPS)
 * - Each frame is seeked to, extracted, landmarks computed, then displayed
 * - Rolling buffer of 30 frames for inference
 * - Prediction runs every N frames for performance
 * - Video playback speed is determined by processing speed (like Python cv2.read())
 */

"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import {
  runInference,
  preloadModel,
  isModelLoaded,
  type InferenceResult,
} from "@/lib/inference";
import {
  preloadLandmarkers,
  areLandmarkersReady,
  extractLandmarksWithDrawingData,
  drawLandmarks,
} from "@/lib/landmarks";

// Configuration matching Python demo
const CONFIG = {
  SEQUENCE_LENGTH: 30,
  CONFIDENCE_THRESHOLD: 0.5,
  SHOW_TOP_K: 5,
  PRED_EVERY_N_FRAMES: 2, // Predict every N frames for performance
  TARGET_FPS: 30, // Target frame extraction rate
};

interface VideoInferencePlayerProps {
  videoBlob: Blob;
  filename?: string;
  onComplete?: (result: InferenceResult) => void;
}

interface InferenceState {
  prediction: string;
  confidence: number;
  topK: Array<{ gloss: string; confidence: number }>;
  // Best prediction seen across all frames
  bestPrediction: string;
  bestConfidence: number;
  bestTopK: Array<{ gloss: string; confidence: number }>;
  frameId: number;
  totalFrames: number;
  bufferSize: number;
  fps: number;
  isProcessing: boolean;
  handsDetected: boolean;
}

export function VideoInferencePlayer({ 
  videoBlob, 
  filename,
  onComplete 
}: VideoInferencePlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const displayCanvasRef = useRef<HTMLCanvasElement>(null);
  const processingCanvasRef = useRef<HTMLCanvasElement>(null);
  const frameBufferRef = useRef<Float32Array[]>([]);
  const processingRef = useRef(false);
  const pausedRef = useRef(false);
  const lastFrameTimeRef = useRef<number>(0);
  const fpsHistoryRef = useRef<number[]>([]);
  const lastResultRef = useRef<InferenceResult | null>(null);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [inferenceState, setInferenceState] = useState<InferenceState>({
    prediction: "...",
    confidence: 0,
    topK: [],
    bestPrediction: "...",
    bestConfidence: 0,
    bestTopK: [],
    frameId: 0,
    totalFrames: 0,
    bufferSize: 0,
    fps: 0,
    isProcessing: false,
    handsDetected: false,
  });

  // Initialize models
  useEffect(() => {
    const init = async () => {
      try {
        if (!isModelLoaded()) {
          setLoadingProgress("Loading ONNX model...");
          await preloadModel();
        }
        if (!areLandmarkersReady()) {
          setLoadingProgress("Loading MediaPipe...");
          await preloadLandmarkers();
        }
        setIsReady(true);
        setLoadingProgress("");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load models");
      } finally {
        setIsLoading(false);
      }
    };
    init();
  }, []);

  // Create video URL from blob
  useEffect(() => {
    const url = URL.createObjectURL(videoBlob);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [videoBlob]);

  // Set up video metadata
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoUrl) return;

    const handleMetadata = () => {
      // Get actual FPS from video or default to 30
      const totalFrames = Math.floor(video.duration * CONFIG.TARGET_FPS);
      setInferenceState(prev => ({ ...prev, totalFrames }));

      // Set canvas sizes
      if (displayCanvasRef.current) {
        displayCanvasRef.current.width = video.videoWidth;
        displayCanvasRef.current.height = video.videoHeight;
      }
      if (processingCanvasRef.current) {
        processingCanvasRef.current.width = video.videoWidth;
        processingCanvasRef.current.height = video.videoHeight;
      }
    };

    video.addEventListener("loadedmetadata", handleMetadata);
    return () => video.removeEventListener("loadedmetadata", handleMetadata);
  }, [videoUrl]);

  // Check if hands are detected
  const checkHandsDetected = useCallback((landmarks: Float32Array): boolean => {
    const POSE_FEATURES = 132;
    const HAND_FEATURES = 63;
    let leftSum = 0, rightSum = 0;
    for (let i = 0; i < HAND_FEATURES; i++) {
      leftSum += Math.abs(landmarks[POSE_FEATURES + i]);
      rightSum += Math.abs(landmarks[POSE_FEATURES + HAND_FEATURES + i]);
    }
    return leftSum > 0.01 || rightSum > 0.01;
  }, []);

  // Main processing loop - processes frames sequentially like Python baseline
  const runProcessingLoop = useCallback(async () => {
    const video = videoRef.current;
    const displayCanvas = displayCanvasRef.current;
    const processingCanvas = processingCanvasRef.current;

    if (!video || !displayCanvas || !processingCanvas) return;

    const displayCtx = displayCanvas.getContext("2d");
    const processingCtx = processingCanvas.getContext("2d");
    if (!displayCtx || !processingCtx) return;

    const totalFrames = Math.floor(video.duration * CONFIG.TARGET_FPS);
    const frameDuration = 1 / CONFIG.TARGET_FPS;

    processingRef.current = true;
    setIsPlaying(true);
    frameBufferRef.current = [];
    fpsHistoryRef.current = [];
    lastFrameTimeRef.current = performance.now();

    // Initialize prediction state OUTSIDE the loop so values persist across frames
    let prediction = "...";
    let confidence = 0;
    let topK: Array<{ gloss: string; confidence: number }> = [];
    
    // Track the BEST prediction across all frames (highest confidence non-unknown)
    let bestPrediction = "...";
    let bestConfidence = 0;
    let bestTopK: Array<{ gloss: string; confidence: number }> = [];

    // Process each frame sequentially (like Python's while cap.read() loop)
    for (let frameId = 0; frameId < totalFrames && processingRef.current; frameId++) {
      // Check if paused
      while (pausedRef.current && processingRef.current) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      if (!processingRef.current) break;

      // Seek to frame time
      const targetTime = frameId * frameDuration;
      video.currentTime = targetTime;

      // Wait for seek to complete
      await new Promise<void>((resolve) => {
        const onSeeked = () => {
          video.removeEventListener("seeked", onSeeked);
          resolve();
        };
        // If already at the right time, resolve immediately
        if (Math.abs(video.currentTime - targetTime) < 0.01) {
          resolve();
        } else {
          video.addEventListener("seeked", onSeeked);
        }
      });

      // Calculate FPS
      const now = performance.now();
      const dt = now - lastFrameTimeRef.current;
      const instantFps = dt > 0 ? 1000 / dt : 0;
      fpsHistoryRef.current.push(instantFps);
      if (fpsHistoryRef.current.length > 10) {
        fpsHistoryRef.current.shift();
      }
      lastFrameTimeRef.current = now;
      const avgFps = fpsHistoryRef.current.reduce((a, b) => a + b, 0) / fpsHistoryRef.current.length;

      // Draw frame to processing canvas
      processingCtx.drawImage(video, 0, 0, processingCanvas.width, processingCanvas.height);

      try {
        // Extract landmarks
        const { features: landmarks, drawingData } = await extractLandmarksWithDrawingData(processingCanvas);
        const handsDetected = checkHandsDetected(landmarks);

        // Draw frame to display canvas (the one user sees)
        displayCtx.drawImage(video, 0, 0, displayCanvas.width, displayCanvas.height);

        // Draw skeleton overlay on display canvas
        if (showSkeleton) {
          drawLandmarks(displayCtx, drawingData, displayCanvas.width, displayCanvas.height, {
            showPose: true,
            showHands: true,
            poseColor: "rgba(0, 255, 0, 0.7)",
            leftHandColor: "rgba(255, 0, 0, 0.9)",
            rightHandColor: "rgba(0, 100, 255, 0.9)",
            lineWidth: 3,
            pointRadius: 5,
          });
        }

        // Add to rolling buffer
        frameBufferRef.current.push(landmarks);
        if (frameBufferRef.current.length > CONFIG.SEQUENCE_LENGTH) {
          frameBufferRef.current.shift();
        }

        // Run inference when buffer is full and every N frames
        // (prediction, confidence, topK are declared outside the loop to persist)
        if (frameBufferRef.current.length === CONFIG.SEQUENCE_LENGTH && 
            frameId % CONFIG.PRED_EVERY_N_FRAMES === 0) {
          
          // Flatten buffer
          const flatLandmarks = new Float32Array(CONFIG.SEQUENCE_LENGTH * 258);
          for (let i = 0; i < CONFIG.SEQUENCE_LENGTH; i++) {
            flatLandmarks.set(frameBufferRef.current[i], i * 258);
          }

          const result = await runInference(flatLandmarks, CONFIG.SHOW_TOP_K, CONFIG.CONFIDENCE_THRESHOLD);
          lastResultRef.current = result;

          prediction = result.gloss;
          confidence = result.confidence;
          topK = result.topK.map(r => ({ gloss: r.gloss, confidence: r.confidence }));

          // Track the BEST prediction (highest confidence that's not "unknown")
          // This ensures we capture the peak prediction even if video ends with empty frames
          if (result.gloss !== "unknown" && result.confidence > bestConfidence) {
            bestPrediction = result.gloss;
            bestConfidence = result.confidence;
            bestTopK = [...topK];
          }

          // Log prediction like Python baseline
          console.log(`[frame=${frameId}] Pred: ${prediction} | prob=${confidence.toFixed(4)} | best: ${bestPrediction} (${bestConfidence.toFixed(4)})`);
        }

        // Update state
        setInferenceState({
          prediction,
          confidence,
          topK,
          bestPrediction,
          bestConfidence,
          bestTopK,
          frameId,
          totalFrames,
          bufferSize: frameBufferRef.current.length,
          fps: avgFps,
          isProcessing: true,
          handsDetected,
        });

      } catch (err) {
        console.error("Frame processing error:", err);
      }

      // Small delay to allow UI updates and prevent blocking
      await new Promise(resolve => setTimeout(resolve, 1));
    }

    // Processing complete
    processingRef.current = false;
    setIsPlaying(false);
    setInferenceState(prev => ({ ...prev, isProcessing: false }));

    // Call onComplete with last result
    if (lastResultRef.current) {
      onComplete?.(lastResultRef.current);
    }
  }, [checkHandsDetected, showSkeleton, onComplete]);

  // Start processing
  const handleStart = useCallback(() => {
    if (!isReady) return;
    pausedRef.current = false;
    setIsPaused(false);
    setInferenceState(prev => ({
      ...prev,
      prediction: "...",
      confidence: 0,
      topK: [],
      bestPrediction: "...",
      bestConfidence: 0,
      bestTopK: [],
      frameId: 0,
      bufferSize: 0,
      isProcessing: true,
    }));
    runProcessingLoop();
  }, [isReady, runProcessingLoop]);

  // Pause/Resume
  const handlePauseResume = useCallback(() => {
    if (pausedRef.current) {
      pausedRef.current = false;
      setIsPaused(false);
    } else {
      pausedRef.current = true;
      setIsPaused(true);
    }
  }, []);

  // Stop
  const handleStop = useCallback(() => {
    processingRef.current = false;
    pausedRef.current = false;
    setIsPlaying(false);
    setIsPaused(false);
    frameBufferRef.current = [];
    setInferenceState(prev => ({
      ...prev,
      frameId: 0,
      bufferSize: 0,
      isProcessing: false,
    }));
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      processingRef.current = false;
    };
  }, []);

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-700 dark:text-red-300">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Loading State */}
      {isLoading && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-500 border-t-transparent"></div>
            <p className="text-blue-700 dark:text-blue-300 text-sm">{loadingProgress || "Loading..."}</p>
          </div>
        </div>
      )}

      {/* Video Container */}
      <div className="relative rounded-xl overflow-hidden bg-gray-900 aspect-video">
        {/* Hidden video element (source only) */}
        <video
          ref={videoRef}
          src={videoUrl || undefined}
          className="hidden"
          playsInline
          muted
          preload="auto"
        />

        {/* Hidden processing canvas */}
        <canvas ref={processingCanvasRef} className="hidden" />

        {/* Display canvas (what user sees) */}
        <canvas
          ref={displayCanvasRef}
          className="w-full h-full object-contain"
        />

        {/* Top-Left: Status and Prediction Overlay */}
        <div className="absolute top-4 left-4 space-y-2 max-w-[60%]">
          {/* Filename/GT Label */}
          {filename && (
            <div className="bg-green-700/80 text-white px-3 py-1.5 rounded text-sm font-medium backdrop-blur-sm">
              GT: {filename.replace(/\.[^/.]+$/, "").split(/[_/\\]/).pop()}
            </div>
          )}

          {/* Best Prediction (main display) */}
          <div className="bg-blue-800/90 text-white px-3 py-2 rounded backdrop-blur-sm border-2 border-blue-400">
            <div className="text-xs opacity-75 uppercase tracking-wide">Best Prediction</div>
            <div className="font-bold text-xl">
              {inferenceState.bestPrediction}
            </div>
            <div className="text-sm opacity-90">
              prob={inferenceState.bestConfidence.toFixed(3)}
            </div>
          </div>

          {/* Current Frame Prediction (secondary) */}
          <div className="bg-gray-800/70 text-white px-3 py-1.5 rounded backdrop-blur-sm text-sm">
            <span className="opacity-75">Current: </span>
            <span className="font-medium">{inferenceState.prediction}</span>
            <span className="opacity-75"> ({(inferenceState.confidence * 100).toFixed(1)}%)</span>
          </div>

          {/* Top-K Predictions from Best */}
          {inferenceState.bestTopK.length > 1 && (
            <div className="bg-black/60 text-white px-3 py-2 rounded backdrop-blur-sm space-y-1">
              <div className="text-xs opacity-75 uppercase tracking-wide">Top-K (at best frame)</div>
              {inferenceState.bestTopK.slice(0, CONFIG.SHOW_TOP_K).map((pred, idx) => (
                <div key={idx} className="text-sm">
                  {idx + 1}. {pred.gloss} ({(pred.confidence * 100).toFixed(1)}%)
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Bottom-Right: Status Indicators */}
        <div className="absolute bottom-14 right-4 text-right space-y-1">
          <div className={`text-sm ${inferenceState.handsDetected ? "text-green-400" : "text-red-400"}`}>
            Hands: {inferenceState.handsDetected ? "DETECTED" : "NOT DETECTED"}
          </div>
        </div>

        {/* Bottom: Status Bar */}
        <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-white px-4 py-2 text-sm backdrop-blur-sm">
          <div className="flex justify-between items-center">
            <span>
              frame: {inferenceState.frameId}/{inferenceState.totalFrames}
            </span>
            <span>
              buffer: {inferenceState.bufferSize}/{CONFIG.SEQUENCE_LENGTH}
            </span>
            <span>
              FPS: {inferenceState.fps.toFixed(1)}
            </span>
          </div>
        </div>

        {/* Buffer Progress Bar */}
        <div className="absolute bottom-10 left-0 right-0 h-1 bg-gray-700">
          <div
            className="h-full bg-green-500 transition-all duration-100"
            style={{ width: `${(inferenceState.bufferSize / CONFIG.SEQUENCE_LENGTH) * 100}%` }}
          />
        </div>

        {/* Frame Progress Bar */}
        <div className="absolute bottom-11 left-0 right-0 h-1 bg-gray-600">
          <div
            className="h-full bg-blue-500 transition-all duration-100"
            style={{ width: `${inferenceState.totalFrames > 0 ? (inferenceState.frameId / inferenceState.totalFrames) * 100 : 0}%` }}
          />
        </div>

        {/* Paused Overlay */}
        {isPaused && (
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <div className="bg-white/80 text-black px-6 py-3 rounded-lg text-lg font-medium backdrop-blur-sm">
              PAUSED (click Resume to continue)
            </div>
          </div>
        )}

        {/* Play Prompt (when not playing) */}
        {!isPlaying && !isLoading && isReady && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40">
            <button
              onClick={handleStart}
              className="bg-green-600 hover:bg-green-700 text-white px-8 py-4 rounded-xl text-lg font-medium transition-colors flex items-center space-x-3"
            >
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
              <span>Run Inference Demo</span>
            </button>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-3 justify-center">
        {isPlaying ? (
          <>
            <button
              onClick={handlePauseResume}
              className={`px-6 py-3 font-medium rounded-lg transition-colors flex items-center space-x-2 ${
                isPaused
                  ? "bg-green-600 text-white hover:bg-green-700"
                  : "bg-yellow-600 text-white hover:bg-yellow-700"
              }`}
            >
              {isPaused ? (
                <>
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                  <span>Resume</span>
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                  </svg>
                  <span>Pause</span>
                </>
              )}
            </button>
            <button
              onClick={handleStop}
              className="px-6 py-3 bg-red-600 text-white font-medium rounded-lg hover:bg-red-700 transition-colors flex items-center space-x-2"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" />
              </svg>
              <span>Stop</span>
            </button>
          </>
        ) : (
          <button
            onClick={handleStart}
            disabled={!isReady}
            className="px-6 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
            <span>Run Inference Demo</span>
          </button>
        )}

        {/* Skeleton Toggle */}
        <button
          onClick={() => setShowSkeleton(!showSkeleton)}
          className={`px-4 py-3 font-medium rounded-lg transition-colors text-sm flex items-center space-x-2 ${
            showSkeleton
              ? "bg-purple-600 text-white hover:bg-purple-700"
              : "bg-gray-600 text-white hover:bg-gray-700"
          }`}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
          <span>{showSkeleton ? "Hide Skeleton" : "Show Skeleton"}</span>
        </button>
      </div>

      {/* Instructions */}
      <div className="text-center text-sm text-gray-500 dark:text-gray-400 space-y-1">
        <p className="font-medium">Video Inference Demo</p>
        <p>Click &quot;Run Inference Demo&quot; to process video frame-by-frame with real-time predictions</p>
        <p>Video plays at processing speed (like Python baseline) - not at native FPS</p>
      </div>
    </div>
  );
}
