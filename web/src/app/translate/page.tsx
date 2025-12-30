"use client";

import { useState, useCallback } from "react";
import { VideoUpload } from "@/components/VideoUpload";
import { WebcamCapture } from "@/components/WebcamCapture";
import { TranslationResult } from "@/components/TranslationResult";
import { predictFromVideo, PredictionResult } from "@/lib/api";

type InputMode = "upload" | "webcam";

export default function TranslatePage() {
  const [mode, setMode] = useState<InputMode>("upload");
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const clearResults = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  const handleVideoSubmit = useCallback(async (videoBlob: Blob, filename?: string) => {
    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      const file = new File([videoBlob], filename || "recording.webm", {
        type: videoBlob.type || "video/webm",
      });
      const prediction = await predictFromVideo(file);
      setResult(prediction);

      // Save to history
      saveToHistory(prediction, filename || "Webcam Recording");
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const saveToHistory = (prediction: PredictionResult, source: string) => {
    try {
      const history = JSON.parse(localStorage.getItem("translationHistory") || "[]");
      history.unshift({
        id: Date.now(),
        timestamp: new Date().toISOString(),
        source,
        prediction: prediction.predicted_gloss,
        confidence: prediction.confidence,
        top5: prediction.top_5_predictions,
      });
      // Keep only last 50 items
      localStorage.setItem("translationHistory", JSON.stringify(history.slice(0, 50)));
    } catch {
      // Ignore storage errors
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Translate Sign Language
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            Upload a video or use your webcam to translate Malaysian Sign Language
          </p>
        </div>

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
          {mode === "upload" ? (
            <VideoUpload onSubmit={handleVideoSubmit} onFileChange={clearResults} isProcessing={isProcessing} />
          ) : (
            <WebcamCapture onCapture={handleVideoSubmit} onRecordStart={clearResults} isProcessing={isProcessing} />
          )}
        </div>

        {/* Processing Indicator */}
        {isProcessing && (
          <div className="flex flex-col items-center justify-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-200 border-t-primary-600 mb-4"></div>
            <p className="text-gray-600 dark:text-gray-300">Processing video...</p>
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
