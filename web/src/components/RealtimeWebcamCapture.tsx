/**
 * Real-time webcam capture component with continuous inference.
 * 
 * Implements optimizations from webcam_demo.py:
 * - Continuous frame processing
 * - Movement detection
 * - Prediction stability filter
 * - Hand detection check
 */

"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { useRealtimeInference, RealtimePrediction } from "@/hooks/useRealtimeInference";

interface RealtimeWebcamCaptureProps {
  onPrediction?: (prediction: RealtimePrediction) => void;
}

export function RealtimeWebcamCapture({ onPrediction }: RealtimeWebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDebug, setShowDebug] = useState(false);

  const {
    isReady,
    isLoading,
    loadingProgress,
    error: inferenceError,
    isProcessing,
    prediction,
    initialize,
    startProcessing,
    stopProcessing,
    reset,
    config,
    showSkeleton,
    setShowSkeleton,
  } = useRealtimeInference();

  // Notify parent of predictions
  useEffect(() => {
    if (prediction && onPrediction) {
      onPrediction(prediction);
    }
  }, [prediction, onPrediction]);

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
        
        // Wait for video to be ready
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

  // Stop inference
  const handleStop = useCallback(() => {
    stopProcessing();
  }, [stopProcessing]);

  // Reset everything
  const handleReset = useCallback(() => {
    stopProcessing();
    reset();
  }, [stopProcessing, reset]);

  // Auto-start webcam
  useEffect(() => {
    startStream();
    return () => {
      stopStream();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Get status color and text
  const getStatusInfo = useCallback(() => {
    if (!prediction) return { color: "bg-gray-500", text: "Waiting..." };
    
    if (prediction.isStable && prediction.confidence >= config.CONFIDENCE_THRESHOLD) {
      return { color: "bg-green-500", text: "STABLE" };
    }
    if (prediction.handsDetected && prediction.isMoving) {
      return { color: "bg-yellow-500", text: "DETECTING" };
    }
    if (prediction.handsDetected) {
      return { color: "bg-blue-500", text: "HANDS VISIBLE" };
    }
    return { color: "bg-gray-500", text: "NO HANDS" };
  }, [prediction, config.CONFIDENCE_THRESHOLD]);

  const statusInfo = getStatusInfo();

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

      {/* Video Display */}
      <div className="relative rounded-xl overflow-hidden bg-gray-900 aspect-video">
        {!isStreaming && !error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-gray-400 border-t-white mb-4"></div>
            <p className="text-gray-400">Starting camera...</p>
          </div>
        )}

        {/* Live video feed (mirrored for selfie view) */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Overlay canvas for skeleton drawing (mirrored to match video) */}
        <canvas 
          ref={overlayCanvasRef} 
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Hidden canvas for processing */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Top-left: Status and prediction */}
        <div className="absolute top-4 left-4 space-y-2">
          {/* Status Badge */}
          <div className={`flex items-center space-x-2 ${statusInfo.color} text-white px-3 py-1.5 rounded-full text-sm font-medium`}>
            <span className={`w-2 h-2 rounded-full bg-white ${isProcessing ? "animate-pulse" : ""}`}></span>
            <span>{statusInfo.text}</span>
          </div>

          {/* Prediction Display */}
          {prediction && prediction.isStable && prediction.confidence >= config.CONFIDENCE_THRESHOLD && (
            <div className="bg-black/70 backdrop-blur-sm text-white px-4 py-3 rounded-lg">
              <p className="text-2xl font-bold">{prediction.gloss}</p>
              <p className="text-sm text-gray-300">
                {(prediction.confidence * 100).toFixed(1)}% confidence
              </p>
            </div>
          )}
        </div>

        {/* Bottom-right: Status Indicators */}
        <div className="absolute bottom-4 right-4 space-y-1 text-right">
          <div className={`text-sm ${prediction?.handsDetected ? "text-green-400" : "text-red-400"}`}>
            Hands: {prediction?.handsDetected ? "DETECTED" : "NOT DETECTED"}
          </div>
          <div className={`text-sm ${prediction?.isMoving ? "text-green-400" : "text-gray-400"}`}>
            Moving: {prediction?.isMoving ? "YES" : "NO"}
          </div>
          <div className="text-sm text-white">
            Buffer: {Math.round((prediction?.bufferProgress || 0) * 100)}%
          </div>
        </div>

        {/* Buffer Progress Bar */}
        {isProcessing && (
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-gray-700">
            <div 
              className="h-full bg-green-500 transition-all duration-100"
              style={{ width: `${(prediction?.bufferProgress || 0) * 100}%` }}
            />
          </div>
        )}
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
            <span>Start Real-time Detection</span>
          </button>
        )}

        {isProcessing && (
          <>
            <button
              onClick={handleStop}
              className="px-6 py-3 bg-red-600 text-white font-medium rounded-lg hover:bg-red-700 transition-colors flex items-center space-x-2"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
              <span>Stop Detection</span>
            </button>
            <button
              onClick={handleReset}
              className="px-6 py-3 bg-gray-600 text-white font-medium rounded-lg hover:bg-gray-700 transition-colors"
            >
              Reset
            </button>
          </>
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

        {/* Debug Toggle */}
        <button
          onClick={() => setShowDebug(!showDebug)}
          className="px-4 py-3 bg-gray-700 text-white font-medium rounded-lg hover:bg-gray-800 transition-colors text-sm"
        >
          {showDebug ? "Hide Debug" : "Show Debug"}
        </button>
      </div>

      {/* Debug Panel */}
      {showDebug && (
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 text-sm font-mono space-y-1">
          <p><strong>Configuration:</strong></p>
          <p className="ml-4">Confidence Threshold: {config.CONFIDENCE_THRESHOLD}</p>
          <p className="ml-4">Sequence Length: {config.SEQUENCE_LENGTH} frames</p>
          <p className="ml-4">Stability Required: {config.STABLE_COUNT_REQUIRED} predictions</p>
          <p className="ml-4">Movement Threshold: {config.MOVEMENT_THRESHOLD}</p>
          <p className="mt-2"><strong>State:</strong></p>
          <p className="ml-4">Model Ready: {isReady ? "Yes" : "No"}</p>
          <p className="ml-4">Processing: {isProcessing ? "Yes" : "No"}</p>
          <p className="ml-4">Buffer Progress: {((prediction?.bufferProgress || 0) * 100).toFixed(1)}%</p>
          <p className="ml-4">Current Prediction: {prediction?.gloss || "None"}</p>
          <p className="ml-4">Is Stable: {prediction?.isStable ? "Yes" : "No"}</p>
        </div>
      )}

      {/* Instructions */}
      <div className="text-center text-sm text-gray-500 dark:text-gray-400 space-y-1">
        <p className="font-medium">Real-time Mode</p>
        <p>Position yourself with good lighting</p>
        <p>Make sure your hands are visible in the frame</p>
        <p>Perform signs naturally - prediction appears when stable</p>
      </div>
    </div>
  );
}
