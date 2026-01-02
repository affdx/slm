/**
 * Real-time webcam capture component with improved UX.
 *
 * Features:
 * - Clear visual state indicators
 * - Horizontal pill list for recent signs at top
 * - Distance feedback (too far/too close)
 * - Dynamic instructions always visible
 * - Circular progress ring
 */

"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import {
  useRealtimeInference,
  type DetectionState,
  type DetectedSign,
  type DistanceStatus,
} from "@/hooks/useRealtimeInference";

interface RealtimeWebcamCaptureProps {
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
    label: "Recording",
    instruction: "Keep signing! Collecting frames...",
  },
  ANALYZING: {
    color: "text-purple-400",
    bgColor: "bg-purple-500/20",
    borderColor: "border-purple-500",
    label: "Analyzing",
    instruction: "Processing your sign...",
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
    >
      {sign.gloss}
      <span className="ml-1 text-xs opacity-70">
        {(sign.confidence * 100).toFixed(0)}%
      </span>
    </div>
  );
}

export function RealtimeWebcamCapture({ onPrediction }: RealtimeWebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
  } = useRealtimeInference();

  // Notify parent of new detections
  useEffect(() => {
    if (currentSign && onPrediction) {
      onPrediction(currentSign);
    }
  }, [currentSign, onPrediction]);

  // Stop webcam stream
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

  // Start webcam stream
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

  // Start real-time inference
  const handleStart = useCallback(() => {
    if (videoRef.current && canvasRef.current && isReady) {
      reset();
      startProcessing(videoRef.current, canvasRef.current, overlayCanvasRef.current || undefined);
    }
  }, [isReady, reset, startProcessing]);

  // Auto-start webcam
  useEffect(() => {
    startStream();
    return () => {
      stopStream();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stateConfig = STATE_CONFIG[detectionState];
  const distanceWarning = prediction?.distanceStatus ? DISTANCE_CONFIG[prediction.distanceStatus] : null;

  return (
    <div className="space-y-4">
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

      {/* Sign History Pills - Always visible to prevent layout shift */}
      <div className="bg-gray-900/80 backdrop-blur-sm rounded-lg p-3 min-h-[88px]">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-gray-400 font-medium">Detected Signs:</span>
          {signHistory.length > 0 && (
            <button
              onClick={clearHistory}
              className="text-xs text-gray-500 hover:text-white transition-colors"
            >
              Clear
            </button>
          )}
        </div>
        {signHistory.length > 0 ? (
          <>
            <div className="flex flex-wrap gap-2">
              {signHistory.slice(0, 8).map((sign, index) => (
                <SignPill key={sign.timestamp} sign={sign} isLatest={index === 0} />
              ))}
            </div>
            {/* Sentence preview */}
            <div className="mt-3 pt-2 border-t border-gray-700">
              <p className="text-sm text-gray-300">
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

      {/* Main Video Area */}
      <div className="relative rounded-xl overflow-hidden bg-gray-900 aspect-video">
        {!isStreaming && !error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white z-10">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-gray-400 border-t-white mb-4"></div>
            <p className="text-gray-400">Starting camera...</p>
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

        {/* State Badge - Top Left */}
        {isProcessing && (
          <div className="absolute top-4 left-4">
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${stateConfig.bgColor} border ${stateConfig.borderColor}`}>
              {detectionState === "COLLECTING" && (
                <div className="relative w-3 h-3">
                  <div className="absolute inset-0 rounded-full bg-red-500 animate-ping opacity-75"></div>
                  <div className="relative rounded-full w-3 h-3 bg-red-500"></div>
                </div>
              )}
              {detectionState === "ANALYZING" && (
                <div className="w-3 h-3 border-2 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
              )}
              {detectionState === "DETECTED" && (
                <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
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

        {/* Current Prediction Display - Show during ANALYZING */}
        {isProcessing && detectionState === "ANALYZING" && prediction && prediction.gloss !== "..." && (
          <div className="absolute top-16 left-1/2 transform -translate-x-1/2 z-10">
            <div className="bg-purple-600/90 text-white px-6 py-3 rounded-xl text-center shadow-lg">
              <p className="text-2xl font-bold">{prediction.gloss}</p>
              <p className="text-purple-200 text-sm">
                {(prediction.confidence * 100).toFixed(0)}% analyzing...
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
        {isProcessing && detectionState === "COLLECTING" && (
          <div className="absolute bottom-4 left-4">
            <CircularProgress
              progress={prediction?.bufferProgress || 0}
              size={56}
              strokeWidth={4}
              color="text-blue-400"
            />
          </div>
        )}

        {/* Hand & Movement Status - Bottom Right */}
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
            {prediction?.handsDetected && (
              <div className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium ${
                prediction?.isMoving 
                  ? "bg-blue-500/20 text-blue-400" 
                  : "bg-gray-500/20 text-gray-400"
              }`}>
                <span className={`w-2 h-2 rounded-full ${prediction?.isMoving ? "bg-blue-400 animate-pulse" : "bg-gray-400"}`}></span>
                {prediction?.isMoving ? "Moving" : "Still"}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Dynamic Instructions - Always visible during processing */}
      <div className={`p-4 rounded-lg border ${stateConfig.bgColor} ${stateConfig.borderColor}`}>
        <div className="flex items-center gap-3">
          <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${stateConfig.bgColor} border ${stateConfig.borderColor}`}>
            {detectionState === "IDLE" && (
              <svg className={`w-5 h-5 ${stateConfig.color}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              </svg>
            )}
            {detectionState === "WAITING_FOR_HANDS" && (
              <svg className={`w-5 h-5 ${stateConfig.color}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
              </svg>
            )}
            {detectionState === "COLLECTING" && (
              <div className="relative w-5 h-5">
                <div className="absolute inset-0 rounded-full bg-red-500 animate-ping opacity-50"></div>
                <div className="relative rounded-full w-5 h-5 bg-red-500"></div>
              </div>
            )}
            {detectionState === "ANALYZING" && (
              <svg className={`w-5 h-5 ${stateConfig.color} animate-spin`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            )}
            {detectionState === "DETECTED" && (
              <svg className={`w-5 h-5 ${stateConfig.color}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            )}
            {detectionState === "COOLDOWN" && (
              <svg className={`w-5 h-5 ${stateConfig.color}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
          </div>
          <div className="flex-1">
            <p className={`font-semibold ${stateConfig.color}`}>{stateConfig.label}</p>
            <p className="text-sm text-gray-400">{stateConfig.instruction}</p>
          </div>
          {isProcessing && detectionState === "COLLECTING" && (
            <div className="flex-shrink-0">
              <CircularProgress
                progress={prediction?.bufferProgress || 0}
                size={48}
                strokeWidth={4}
                color="text-blue-400"
              />
            </div>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-3 justify-center">
        {isStreaming && !isProcessing && (
          <button
            onClick={handleStart}
            disabled={!isReady || isLoading}
            className="px-6 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Start Detection</span>
          </button>
        )}

        {isProcessing && (
          <>
            <button
              onClick={stopProcessing}
              className="px-6 py-3 bg-red-600 text-white font-medium rounded-lg hover:bg-red-700 transition-colors flex items-center space-x-2"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
              <span>Stop</span>
            </button>
            <button
              onClick={reset}
              className="px-4 py-3 bg-gray-600 text-white font-medium rounded-lg hover:bg-gray-700 transition-colors"
            >
              Reset
            </button>
          </>
        )}

        <button
          onClick={() => setShowSkeleton(!showSkeleton)}
          className={`px-4 py-3 font-medium rounded-lg transition-colors text-sm ${
            showSkeleton
              ? "bg-purple-600 text-white hover:bg-purple-700"
              : "bg-gray-600 text-white hover:bg-gray-700"
          }`}
        >
          {showSkeleton ? "Hide Skeleton" : "Show Skeleton"}
        </button>
      </div>

      {/* Help Section - Only when not processing */}
      {!isProcessing && (
        <div className="bg-gray-100 dark:bg-gray-800/50 rounded-lg p-4 text-sm">
          <p className="font-medium text-gray-700 dark:text-gray-300 mb-2">How to use:</p>
          <ol className="text-gray-600 dark:text-gray-400 space-y-1 list-decimal list-inside">
            <li>Click <strong>Start Detection</strong> to begin</li>
            <li>Position yourself at a comfortable distance from the camera</li>
            <li>Show your hands and start signing</li>
            <li>Keep signing until the progress reaches 100%</li>
            <li>See the detected sign appear, then continue with the next sign</li>
          </ol>
        </div>
      )}
    </div>
  );
}
