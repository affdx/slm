"use client";

import { useState, useCallback, useRef } from "react";
import { VideoUpload } from "@/components/VideoUpload";
import { WebcamCapture } from "@/components/WebcamCapture";
import { RealtimeWebcamCapture } from "@/components/RealtimeWebcamCapture";
import { TranslationResult } from "@/components/TranslationResult";
import { ModelLoader } from "@/components/ModelLoader";
import {
  useSignLanguageInference,
  PredictionResult,
} from "@/hooks/useSignLanguageInference";
import { RealtimePrediction } from "@/hooks/useRealtimeInference";
import { addHistoryItem } from "@/lib/history";

type InputMode = "upload" | "webcam" | "realtime";

export default function TranslatePage() {
  const [mode, setMode] = useState<InputMode>("upload");
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [realtimePrediction, setRealtimePrediction] = useState<RealtimePrediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Store the current video blob for history
  const currentVideoBlobRef = useRef<Blob | null>(null);

  // Client-side inference hook
  const { 
    isReady, 
    isLoading, 
    loadingProgress, 
    error: modelError, 
    delegate,
    currentModel,
    availableModels,
    predict,
    switchDelegate,
    switchModel,
  } = useSignLanguageInference();

  const clearResults = useCallback(() => {
    setResult(null);
    setRealtimePrediction(null);
    setError(null);
    currentVideoBlobRef.current = null;
  }, []);

  const handleRealtimePrediction = useCallback((prediction: RealtimePrediction) => {
    setRealtimePrediction(prediction);
  }, []);

  const handleVideoSubmit = useCallback(
    async (videoBlob: Blob, filename?: string) => {
      setIsProcessing(true);
      setError(null);
      setResult(null);

      // Store video blob for history
      currentVideoBlobRef.current = videoBlob;

      try {
        // Use client-side inference
        const prediction = await predict(videoBlob);
        setResult(prediction);

        // Save to history with video
        await addHistoryItem(
          prediction,
          filename || "Webcam Recording",
          videoBlob
        );
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setIsProcessing(false);
      }
    },
    [predict]
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Translate Sign Language
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            Upload a video or use your webcam to translate Malaysian Sign
            Language
          </p>
          {/* Client-side indicator */}
          <div className="mt-2 inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">
            <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></span>
            Client-side inference (no server required)
          </div>

          {/* Model and Processing Controls */}
          {isReady && (
            <div className="mt-4 flex flex-col sm:flex-row items-center justify-center gap-4">
              {/* Model Selector */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500 dark:text-gray-400">Model:</span>
                <select
                  value={currentModel}
                  onChange={(e) => switchModel(e.target.value as typeof currentModel)}
                  disabled={isLoading || isProcessing}
                  className={`
                    text-xs px-3 py-1.5 rounded-md border border-gray-300 dark:border-gray-600
                    bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-200
                    focus:outline-none focus:ring-2 focus:ring-primary-500
                    ${(isLoading || isProcessing) ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
                  `}
                >
                  {availableModels.map(({ type, config }) => (
                    <option key={type} value={type}>
                      {config.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Divider */}
              <div className="hidden sm:block w-px h-4 bg-gray-300 dark:bg-gray-600"></div>

              {/* Processing Mode Toggle */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500 dark:text-gray-400">Processing:</span>
                <button
                  onClick={() => switchDelegate(delegate === "GPU" ? "CPU" : "GPU")}
                  disabled={isLoading || isProcessing}
                  className={`
                    relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                    ${delegate === "GPU" 
                      ? "bg-primary-600" 
                      : "bg-gray-300 dark:bg-gray-600"
                    }
                    ${(isLoading || isProcessing) ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
                  `}
                  title={`Currently using ${delegate}. Click to switch to ${delegate === "GPU" ? "CPU" : "GPU"}.`}
                >
                  <span
                    className={`
                      inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                      ${delegate === "GPU" ? "translate-x-6" : "translate-x-1"}
                    `}
                  />
                </button>
                <span className={`text-xs font-medium ${delegate === "GPU" ? "text-primary-600" : "text-gray-600 dark:text-gray-400"}`}>
                  {delegate}
                </span>
                <span className="text-xs text-gray-400 dark:text-gray-500">
                  ({delegate === "GPU" ? "faster" : "compatible"})
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Model Loading State */}
        {(isLoading || modelError) && (
          <ModelLoader
            isLoading={isLoading}
            progress={loadingProgress}
            error={modelError}
          />
        )}

        {/* Mode Toggle */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex rounded-lg border border-gray-200 dark:border-gray-700 p-1 bg-gray-50 dark:bg-gray-800">
            <button
              onClick={() => { setMode("upload"); clearResults(); }}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                mode === "upload"
                  ? "bg-white dark:bg-gray-700 text-primary-600 shadow-sm"
                  : "text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
              }`}
            >
              Upload Video
            </button>
            <button
              onClick={() => { setMode("webcam"); clearResults(); }}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                mode === "webcam"
                  ? "bg-white dark:bg-gray-700 text-primary-600 shadow-sm"
                  : "text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
              }`}
            >
              Record Webcam
            </button>
            <button
              onClick={() => { setMode("realtime"); clearResults(); }}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-1 ${
                mode === "realtime"
                  ? "bg-white dark:bg-gray-700 text-primary-600 shadow-sm"
                  : "text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
              }`}
            >
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
              </span>
              Real-time
            </button>
          </div>
        </div>

        {/* Input Section */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-8">
          {mode === "upload" && (
            <VideoUpload
              onSubmit={handleVideoSubmit}
              onFileChange={clearResults}
              isProcessing={isProcessing}
              isModelReady={isReady}
            />
          )}
          {mode === "webcam" && (
            <WebcamCapture
              key="webcam"
              onCapture={handleVideoSubmit}
              onRecordStart={clearResults}
              isProcessing={isProcessing}
              isModelReady={isReady}
            />
          )}
          {mode === "realtime" && (
            <RealtimeWebcamCapture
              key="realtime"
              onPrediction={handleRealtimePrediction}
            />
          )}
        </div>

        {/* Processing Indicator */}
        {isProcessing && (
          <div className="flex flex-col items-center justify-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-200 border-t-primary-600 mb-4"></div>
            <p className="text-gray-600 dark:text-gray-300">
              Processing video locally...
            </p>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Extracting landmarks and running inference
            </p>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-8">
            <div className="flex items-center">
              <svg
                className="w-5 h-5 text-red-500 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span className="text-red-700 dark:text-red-300">{error}</span>
            </div>
          </div>
        )}

        {/* Result Display */}
        {result && !isProcessing && mode !== "realtime" && <TranslationResult result={result} />}

        {/* Real-time Mode Info Card */}
        {mode === "realtime" && realtimePrediction && realtimePrediction.isStable && realtimePrediction.confidence >= 0.7 && (
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border border-green-200 dark:border-green-800 rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-green-600 dark:text-green-400 font-medium mb-1">
                  Detected Sign (Stable)
                </p>
                <p className="text-3xl font-bold text-green-800 dark:text-green-200">
                  {realtimePrediction.gloss}
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm text-gray-500 dark:text-gray-400">Confidence</p>
                <p className="text-2xl font-semibold text-green-700 dark:text-green-300">
                  {(realtimePrediction.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
