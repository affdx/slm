"use client";

import { useState, useCallback, useRef } from "react";
import { VideoUpload } from "@/components/VideoUpload";
import { WebcamCapture } from "@/components/WebcamCapture";
import { TranslationResult } from "@/components/TranslationResult";
import { ModelLoader } from "@/components/ModelLoader";
import {
  useSignLanguageInference,
  PredictionResult,
} from "@/hooks/useSignLanguageInference";
import { addHistoryItem } from "@/lib/history";

type InputMode = "upload" | "webcam";

export default function TranslatePage() {
  const [mode, setMode] = useState<InputMode>("upload");
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
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
    predict,
    switchDelegate,
  } = useSignLanguageInference();

  const clearResults = useCallback(() => {
    setResult(null);
    setError(null);
    currentVideoBlobRef.current = null;
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

          {/* Processing Mode Toggle */}
          {isReady && (
            <div className="mt-3 flex items-center justify-center gap-2">
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
              onClick={() => setMode("upload")}
              className={`px-6 py-2 rounded-md text-sm font-medium transition-colors ${
                mode === "upload"
                  ? "bg-white dark:bg-gray-700 text-primary-600 shadow-sm"
                  : "text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
              }`}
            >
              Upload Video
            </button>
            <button
              onClick={() => setMode("webcam")}
              className={`px-6 py-2 rounded-md text-sm font-medium transition-colors ${
                mode === "webcam"
                  ? "bg-white dark:bg-gray-700 text-primary-600 shadow-sm"
                  : "text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
              }`}
            >
              Use Webcam
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
        {result && !isProcessing && <TranslationResult result={result} />}
      </div>
    </div>
  );
}
