/**
 * Real-time webcam capture component V2 with Python-style sliding window.
 *
 * Features:
 * - First-time onboarding overlay with quick-start guide
 * - Sign history pills always visible at top once detection started
 * - Controls (stop, reset, skeleton) inside video frame as icon buttons
 * - Clear visual state indicators
 * - Distance feedback (too far/too close)
 * - Instructions displayed inside video overlay
 * - Circular progress ring
 * - Shows prediction frequency (predictions/sec) for comparison
 */

"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import {
  useRealtimeInferenceV2,
  type DetectionState,
  type DetectedSign,
  type DistanceStatus,
} from "@/hooks/useRealtimeInferenceV2";

interface RealtimeWebcamCaptureV2Props {
  onPrediction?: (sign: DetectedSign) => void;
}

// State configuration for UI
const STATE_CONFIG: Record<DetectionState, {
  color: string;
  bgColor: string;
  borderColor: string;
  label: string;
  instruction: string;
}> = {
  IDLE: {
    color: "text-gray-400",
    bgColor: "bg-gray-500/20",
    borderColor: "border-gray-500",
    label: "Ready",
    instruction: "Click 'Start Detection' to begin translating sign language",
  },
  WAITING_FOR_HANDS: {
    color: "text-yellow-400",
    bgColor: "bg-yellow-500/20",
    borderColor: "border-yellow-500",
    label: "Waiting for Hands",
    instruction: "Show your hands in the camera frame to start",
  },
  COLLECTING: {
    color: "text-blue-400",
    bgColor: "bg-blue-500/20",
    borderColor: "border-blue-500",
    label: "Detecting",
    instruction: "Keep signing! Continuous detection active...",
  },
  DETECTED: {
    color: "text-green-400",
    bgColor: "bg-green-500/20",
    borderColor: "border-green-500",
    label: "Detected!",
    instruction: "Sign recognized! Get ready for the next one.",
  },
  COOLDOWN: {
    color: "text-orange-400",
    bgColor: "bg-orange-500/20",
    borderColor: "border-orange-500",
    label: "Get Ready",
    instruction: "Prepare for the next sign...",
  },
};

// Distance status messages
const DISTANCE_CONFIG: Record<DistanceStatus, { message: string; color: string } | null> = {
  ok: null,
  too_far: { message: "Move closer to camera", color: "text-orange-400 bg-orange-500/20" },
  too_close: { message: "Move back from camera", color: "text-orange-400 bg-orange-500/20" },
  unknown: null,
};

// Circular progress component
function CircularProgress({
  progress,
  size = 60,
  strokeWidth = 5,
  color = "text-blue-400",
}: {
  progress: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
}) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - progress * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="transform -rotate-90" width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          stroke="currentColor"
          fill="none"
          className="text-gray-700"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          stroke="currentColor"
          fill="none"
          strokeLinecap="round"
          className={color}
          style={{
            strokeDasharray: circumference,
            strokeDashoffset: offset,
            transition: "stroke-dashoffset 0.1s ease-out",
          }}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className={`text-xs font-bold ${color}`}>
          {Math.round(progress * 100)}%
        </span>
      </div>
    </div>
  );
}

// Sign pill component
function SignPill({ sign, isLatest }: { sign: DetectedSign; isLatest: boolean }) {
  return (
    <div
      className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
        isLatest
          ? "bg-green-500 text-white scale-110 shadow-lg"
          : "bg-gray-700 text-gray-200"
      }`}
      role="listitem"
      aria-label={`${sign.gloss}, ${(sign.confidence * 100).toFixed(0)}% confidence${isLatest ? ", latest detection" : ""}`}
    >
      {sign.gloss}
      <span className="ml-1 text-xs opacity-70" aria-hidden="true">
        {(sign.confidence * 100).toFixed(0)}%
      </span>
    </div>
  );
}

export function RealtimeWebcamCaptureV2({ onPrediction }: RealtimeWebcamCaptureV2Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasStartedOnce, setHasStartedOnce] = useState(false);

  const {
    isReady,
    isLoading,
    loadingProgress,
    error: inferenceError,
    isProcessing,
    detectionState,
    prediction,
    currentSign,
    signHistory,
    startProcessing,
    stopProcessing,
    reset,
    clearHistory,
    showSkeleton,
    setShowSkeleton,
    config,
  } = useRealtimeInferenceV2();

  useEffect(() => {
    if (currentSign && onPrediction) {
      onPrediction(currentSign);
    }
  }, [currentSign, onPrediction]);

  const stopStream = useCallback(() => {
    stopProcessing();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  }, [stopProcessing]);

  const startStream = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          setIsStreaming(true);
        };
      }
    } catch (err) {
      if (err instanceof Error) {
        if (err.name === "NotAllowedError") {
          setError("Camera access denied. Please allow camera access.");
        } else if (err.name === "NotFoundError") {
          setError("No camera found. Please connect a camera.");
        } else {
          setError(`Failed to access camera: ${err.message}`);
        }
      }
    }
  }, []);

  const handleStart = useCallback(() => {
    if (videoRef.current && canvasRef.current && isReady) {
      reset();
      setHasStartedOnce(true);
      startProcessing(videoRef.current, canvasRef.current, overlayCanvasRef.current || undefined);
    }
  }, [isReady, reset, startProcessing]);

  useEffect(() => {
    startStream();
    return () => {
      stopStream();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!containerRef.current?.contains(document.activeElement) && document.activeElement !== document.body) {
        return;
      }

      switch (event.key.toLowerCase()) {
        case " ":
        case "enter":
          if (!isProcessing && isStreaming && isReady) {
            event.preventDefault();
            handleStart();
          }
          break;
        case "escape":
        case "s":
          if (isProcessing) {
            event.preventDefault();
            stopProcessing();
          }
          break;
        case "r":
          if (isProcessing) {
            event.preventDefault();
            reset();
          }
          break;
        case "k":
          if (isProcessing) {
            event.preventDefault();
            setShowSkeleton(!showSkeleton);
          }
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isProcessing, isStreaming, isReady, handleStart, stopProcessing, reset, showSkeleton, setShowSkeleton]);

  const stateConfig = STATE_CONFIG[detectionState];
  const distanceWarning = prediction?.distanceStatus ? DISTANCE_CONFIG[prediction.distanceStatus] : null;

  return (
    <div className="space-y-4" ref={containerRef} tabIndex={-1}>
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {currentSign && `Detected sign: ${currentSign.gloss} with ${Math.round(currentSign.confidence * 100)}% confidence`}
      </div>
      {/* Error Display */}
      {(error || inferenceError) && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-red-700 dark:text-red-300 text-sm">{error || inferenceError}</p>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-500 border-t-transparent"></div>
            <p className="text-blue-700 dark:text-blue-300 text-sm">{loadingProgress}</p>
          </div>
        </div>
      )}

      {/* V2 Mode Indicator */}
      <div className="bg-gradient-to-r from-purple-500/20 to-blue-500/20 border border-purple-500/50 rounded-lg p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 bg-purple-600 text-white text-xs font-bold rounded">V2</span>
            <span className="text-sm text-purple-300">Python-style Sliding Window</span>
          </div>
          {isProcessing && prediction && (
            <div className="flex items-center gap-4 text-xs text-gray-400">
              <span>Frame: {prediction.frameCount}</span>
              <span className="text-green-400">{prediction.predictionsPerSecond} pred/sec</span>
              <span>Every {config.PRED_EVERY_N_FRAMES} frames</span>
            </div>
          )}
        </div>
      </div>

      {hasStartedOnce && (
        <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg p-3" role="region" aria-label="Detected signs history">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-400 font-medium">Detected Signs:</span>
            {signHistory.length > 0 && (
              <button
                onClick={clearHistory}
                className="text-xs text-gray-500 hover:text-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900 rounded"
                aria-label="Clear sign history"
              >
                Clear
              </button>
            )}
          </div>
          {signHistory.length > 0 ? (
            <>
              <div className="flex flex-wrap gap-2" role="list" aria-label="Recently detected signs">
                {signHistory.slice(0, 8).map((sign, index) => (
                  <SignPill key={sign.timestamp} sign={sign} isLatest={index === 0} />
                ))}
              </div>
              <div className="mt-3 pt-2 border-t border-gray-700">
                <p className="text-sm text-gray-300" aria-label="Composed sentence from detected signs">
                  <span className="text-gray-500">Sentence: </span>
                  {signHistory.slice().reverse().map(s => s.gloss).join(" ")}
                </p>
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-10 text-gray-500 text-sm">
              Signs will appear here as you sign...
            </div>
          )}
        </div>
      )}

      {/* Main Video Area */}
      <div className="relative rounded-xl overflow-hidden bg-gray-900 aspect-video min-h-[400px] md:min-h-0">
        {!isStreaming && !error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white z-10">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-gray-400 border-t-white mb-4"></div>
            <p className="text-gray-400">Starting camera...</p>
          </div>
        )}

        {/* First-time Onboarding Overlay */}
        {isStreaming && !isProcessing && !hasStartedOnce && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-gradient-to-t from-black/90 via-black/60 to-black/40">
            <div className="text-center px-6 max-w-md">
              <div className="mb-6">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-purple-500/20 border-2 border-purple-500 mb-4">
                  <svg className="w-8 h-8 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Live Detection V2</h3>
                <p className="text-gray-300 text-sm mb-2">
                  Python-style sliding window with continuous inference
                </p>
                <p className="text-purple-400 text-xs">
                  ~{Math.round(30 / config.PRED_EVERY_N_FRAMES)} predictions/second • Overlapping windows
                </p>
              </div>

              {/* Quick steps */}
              <div className="grid grid-cols-3 gap-3 mb-6 text-xs">
                <div className="flex flex-col items-center">
                  <div className="w-8 h-8 rounded-full bg-blue-500/20 border border-blue-500 flex items-center justify-center mb-1.5">
                    <span className="text-blue-400 font-bold">1</span>
                  </div>
                  <span className="text-gray-400">Show hands</span>
                </div>
                <div className="flex flex-col items-center">
                  <div className="w-8 h-8 rounded-full bg-blue-500/20 border border-blue-500 flex items-center justify-center mb-1.5">
                    <span className="text-blue-400 font-bold">2</span>
                  </div>
                  <span className="text-gray-400">Start signing</span>
                </div>
                <div className="flex flex-col items-center">
                  <div className="w-8 h-8 rounded-full bg-blue-500/20 border border-blue-500 flex items-center justify-center mb-1.5">
                    <span className="text-blue-400 font-bold">3</span>
                  </div>
                  <span className="text-gray-400">See translation</span>
                </div>
              </div>

              <button
                onClick={handleStart}
                disabled={!isReady || isLoading}
                className="px-8 py-4 bg-purple-600 text-white font-semibold rounded-xl hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 flex items-center justify-center mx-auto space-x-3 shadow-lg shadow-purple-500/30"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>Start V2 Detection</span>
              </button>
            </div>
          </div>
        )}

        {/* Returning User Overlay */}
        {isStreaming && !isProcessing && hasStartedOnce && (
          <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/50">
            <button
              onClick={handleStart}
              disabled={!isReady || isLoading}
              className="px-8 py-4 bg-purple-600 text-white font-semibold rounded-xl hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 flex items-center space-x-3 shadow-lg shadow-purple-500/30"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>Resume V2 Detection</span>
            </button>
          </div>
        )}

        {/* Live video feed (mirrored) */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Overlay canvas for skeleton (mirrored) */}
        <canvas
          ref={overlayCanvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Hidden canvas for processing */}
        <canvas ref={canvasRef} className="hidden" />

        {isProcessing && (
          <div className="absolute top-4 right-4 z-20 flex items-center gap-2" role="group" aria-label="Detection controls">
            <button
              onClick={() => setShowSkeleton(!showSkeleton)}
              className={`p-2 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900 ${
                showSkeleton
                  ? "bg-purple-600 text-white hover:bg-purple-700"
                  : "bg-black/50 text-white hover:bg-black/70"
              }`}
              aria-label={showSkeleton ? "Hide skeleton overlay (K)" : "Show skeleton overlay (K)"}
              aria-pressed={showSkeleton}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </button>
            <button
              onClick={reset}
              className="p-2 rounded-full bg-black/50 text-white hover:bg-black/70 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900"
              aria-label="Reset detection (R)"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
            <button
              onClick={stopProcessing}
              className="p-2 rounded-full bg-red-600 text-white hover:bg-red-700 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900"
              aria-label="Stop detection (S or Escape)"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
            </button>
          </div>
        )}

        {/* State Badge - Top Left */}
        {isProcessing && (
          <div className="absolute top-4 left-4">
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${stateConfig.bgColor} border ${stateConfig.borderColor}`}>
              {detectionState === "COLLECTING" && (
                <div className="relative w-3 h-3">
                  <div className="absolute inset-0 rounded-full bg-blue-500 animate-ping opacity-75"></div>
                  <div className="relative rounded-full w-3 h-3 bg-blue-500"></div>
                </div>
              )}
              {detectionState === "DETECTED" && (
                <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                </svg>
              )}
              {detectionState === "WAITING_FOR_HANDS" && (
                <svg className="w-4 h-4 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
                </svg>
              )}
              {detectionState === "COOLDOWN" && (
                <svg className="w-4 h-4 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )}
              <span className={`text-sm font-medium ${stateConfig.color}`}>
                {stateConfig.label}
              </span>
            </div>
          </div>
        )}

        {/* Distance Warning - Top Center */}
        {isProcessing && distanceWarning && (
          <div className="absolute top-4 left-1/2 transform -translate-x-1/2">
            <div className={`px-4 py-2 rounded-full ${distanceWarning.color} font-medium text-sm`}>
              {distanceWarning.message}
            </div>
          </div>
        )}

        {/* Live Prediction Display - Shows current prediction during COLLECTING */}
        {isProcessing && detectionState === "COLLECTING" && prediction && prediction.gloss !== "..." && (
          <div className="absolute top-16 left-1/2 transform -translate-x-1/2 z-10">
            <div className="bg-blue-600/90 text-white px-6 py-3 rounded-xl text-center shadow-lg">
              <p className="text-2xl font-bold">{prediction.gloss}</p>
              <p className="text-blue-200 text-sm">
                {(prediction.confidence * 100).toFixed(0)}% • Waiting for stability...
              </p>
            </div>
          </div>
        )}

        {/* Detected Sign Display - Center */}
        {detectionState === "DETECTED" && currentSign && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-20">
            <div className="bg-green-500 text-white px-10 py-6 rounded-2xl text-center animate-bounce-in shadow-2xl">
              <p className="text-5xl font-bold mb-2">{currentSign.gloss}</p>
              <p className="text-green-100 text-lg">
                {(currentSign.confidence * 100).toFixed(0)}% confidence
              </p>
            </div>
          </div>
        )}

        {/* Progress Ring - Bottom Left */}
        {isProcessing && detectionState === "COLLECTING" && prediction && prediction.bufferProgress < 1 && (
          <div className="absolute bottom-4 left-4">
            <CircularProgress
              progress={prediction.bufferProgress}
              size={56}
              strokeWidth={4}
              color="text-blue-400"
            />
          </div>
        )}

        {/* Stats - Bottom Left (when buffer is full) */}
        {isProcessing && detectionState === "COLLECTING" && prediction && prediction.bufferProgress >= 1 && (
          <div className="absolute bottom-4 left-4 bg-black/60 rounded-lg px-3 py-2">
            <div className="text-xs text-gray-300 space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-green-400">●</span>
                <span>Buffer Full</span>
              </div>
              <div className="text-purple-400 font-medium">
                {prediction.predictionsPerSecond} pred/sec
              </div>
            </div>
          </div>
        )}

        {/* Hand Status - Bottom Right */}
        {isProcessing && (
          <div className="absolute bottom-4 right-4 flex items-center gap-3">
            <div className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium ${
              prediction?.handsDetected 
                ? "bg-green-500/20 text-green-400" 
                : "bg-red-500/20 text-red-400"
            }`}>
              <span className={`w-2 h-2 rounded-full ${prediction?.handsDetected ? "bg-green-400" : "bg-red-400"}`}></span>
              {prediction?.handsDetected ? "Hands" : "No Hands"}
            </div>
          </div>
        )}

        {/* Instruction Bar - Bottom Center */}
        {isProcessing && (
          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-10">
            <div className={`px-4 py-2 rounded-full ${stateConfig.bgColor} border ${stateConfig.borderColor} backdrop-blur-sm`}>
              <p className={`text-sm font-medium ${stateConfig.color}`}>{stateConfig.instruction}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
