"use client";

import { useState, useCallback, useRef } from "react";
import { VideoUpload } from "@/components/VideoUpload";
import { VideoInferencePlayer } from "@/components/VideoInferencePlayer";
import { WebcamCapture } from "@/components/WebcamCapture";
import { RealtimeWebcamCapture } from "@/components/RealtimeWebcamCapture";
import { TranslationResult } from "@/components/TranslationResult";
import { ModelLoader } from "@/components/ModelLoader";
import {
  useSignLanguageInference,
  PredictionResult,
} from "@/hooks/useSignLanguageInference";
import { DetectedSign } from "@/hooks/useRealtimeInference";
import { addHistoryItem } from "@/lib/history";
import { setModelType } from "@/lib/inference";
import { useEffect } from "react";

type InputMode = "upload" | "webcam" | "realtime";
type UploadSubMode = "quick" | "demo";

export default function TCNTranslatePage() {
  const [mode, setMode] = useState<InputMode>("upload");
  const [uploadSubMode, setUploadSubMode] = useState<UploadSubMode>("quick");
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [realtimePrediction, setRealtimePrediction] = useState<DetectedSign | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const [selectedVideoBlob, setSelectedVideoBlob] = useState<Blob | null>(null);
  const [selectedFilename, setSelectedFilename] = useState<string | null>(null);
  const currentVideoBlobRef = useRef<Blob | null>(null);

  const { 
    isReady, 
    isLoading, 
    loadingProgress, 
    error: modelError, 
    currentModel,
    availableModels,
    predict,
    switchModel,
  } = useSignLanguageInference();

  // Set TCN model on mount
  useEffect(() => {
    setModelType("tcn").catch((err) => {
      console.error("Failed to set TCN model:", err);
      setError("Failed to load TCN model");
    });
  }, []);

  const clearResults = useCallback(() => {
    setResult(null);
    setRealtimePrediction(null);
    setError(null);
    setSelectedVideoBlob(null);
    setSelectedFilename(null);
    currentVideoBlobRef.current = null;
  }, []);

  const handleRealtimePrediction = useCallback((sign: DetectedSign) => {
    setRealtimePrediction(sign);
  }, []);

  const handleVideoSubmit = useCallback(
    async (videoBlob: Blob, filename?: string) => {
      currentVideoBlobRef.current = videoBlob;

      if (mode === "upload" && uploadSubMode === "demo") {
        setSelectedVideoBlob(videoBlob);
        setSelectedFilename(filename || null);
        setError(null);
        setResult(null);
        return;
      }

      setIsProcessing(true);
      setError(null);
      setResult(null);

      try {
        // Ensure TCN model is selected
        await setModelType("tcn");
        const prediction = await predict(videoBlob);
        setResult(prediction);

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
    [predict, mode, uploadSubMode]
  );

  return (
    <div className="min-h-screen bg-gray-50/50 dark:bg-slate-900/50 pt-24 pb-12">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 gap-4">
          <div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white tracking-tight">
                Translate (TCN Model)
              </h1>
              <p className="text-gray-500 dark:text-gray-400 mt-1">
                Real-time Malaysian Sign Language translation using Temporal Convolutional Network
              </p>
              <div className="mt-2">
                <a
                  href="/translate"
                  className="text-xs text-gray-600 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 font-medium"
                >
                  ← Back to LSTM Models
                </a>
              </div>
            </div>
          </div>
          {(isLoading || modelError) && (
            <ModelLoader
              isLoading={isLoading}
              progress={loadingProgress}
              error={modelError}
            />
          )}
        </div>

        {modelError && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl">
            <p className="text-red-800 dark:text-red-200 text-sm">
              {modelError}
            </p>
          </div>
        )}

        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl">
            <p className="text-red-800 dark:text-red-200 text-sm">{error}</p>
          </div>
        )}

        {/* Model Info Banner */}
        <div className="mb-8 p-4 bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 rounded-xl">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-primary-500 flex items-center justify-center text-white font-bold">
              T
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-primary-900 dark:text-primary-100 mb-1">
                Causal Temporal Convolutional Network (TCN)
              </h3>
              <p className="text-sm text-primary-700 dark:text-primary-300">
                This model uses causal dilated convolutions optimized for real-time streaming inference. 
                It processes temporal sequences efficiently and is designed for low-latency applications.
              </p>
            </div>
          </div>
        </div>

        {/* Mode Selection */}
        <div className="mb-8">
          <div className="inline-flex rounded-xl bg-white dark:bg-gray-800 p-1 shadow-sm border border-gray-200 dark:border-gray-700">
            <button
              onClick={() => {
                setMode("upload");
                clearResults();
              }}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                mode === "upload"
                  ? "bg-primary-500 text-white shadow-sm"
                  : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
              }`}
            >
              Upload Video
            </button>
            <button
              onClick={() => {
                setMode("webcam");
                clearResults();
              }}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                mode === "webcam"
                  ? "bg-primary-500 text-white shadow-sm"
                  : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
              }`}
            >
              Record Video
            </button>
            <button
              onClick={() => {
                setMode("realtime");
                clearResults();
              }}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                mode === "realtime"
                  ? "bg-primary-500 text-white shadow-sm"
                  : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
              }`}
            >
              Real-time
            </button>
          </div>
        </div>

        {/* Upload Sub-mode (only for upload mode) */}
        {mode === "upload" && (
          <div className="mb-6">
            <div className="inline-flex rounded-lg bg-gray-100 dark:bg-gray-800 p-1">
              <button
                onClick={() => {
                  setUploadSubMode("quick");
                  clearResults();
                }}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  uploadSubMode === "quick"
                    ? "bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm"
                    : "text-gray-600 dark:text-gray-400"
                }`}
              >
                Quick Translate
              </button>
              <button
                onClick={() => {
                  setUploadSubMode("demo");
                  clearResults();
                }}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  uploadSubMode === "demo"
                    ? "bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm"
                    : "text-gray-600 dark:text-gray-400"
                }`}
              >
                Interactive Demo
              </button>
            </div>
          </div>
        )}

        {/* Main Content Area */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="space-y-6">
            {mode === "upload" && (
              <>
                {uploadSubMode === "quick" ? (
                  <VideoUpload
                    onSubmit={handleVideoSubmit}
                    isProcessing={isProcessing}
                    isModelReady={isReady}
                  />
                ) : (
                  selectedVideoBlob && (
                    <VideoInferencePlayer
                      videoBlob={selectedVideoBlob}
                      filename={selectedFilename || undefined}
                      onComplete={(inferenceResult) => {
                        // Convert InferenceResult to PredictionResult
                        const prediction: PredictionResult = {
                          predicted_gloss: inferenceResult.gloss,
                          confidence: inferenceResult.confidence,
                          top_5_predictions: inferenceResult.topK.map((item) => ({
                            gloss: item.gloss,
                            confidence: item.confidence,
                          })),
                          processing_time_ms: inferenceResult.inferenceTimeMs,
                          landmark_extraction_time_ms: 0,
                          inference_time_ms: inferenceResult.inferenceTimeMs,
                        };
                        setResult(prediction);
                        if (selectedVideoBlob) {
                          addHistoryItem(
                            prediction,
                            selectedFilename || "Uploaded Video",
                            selectedVideoBlob
                          );
                        }
                      }}
                    />
                  )
                )}
              </>
            )}

            {mode === "webcam" && (
              <WebcamCapture
                onCapture={handleVideoSubmit}
                isProcessing={isProcessing}
                isModelReady={isReady}
              />
            )}

            {mode === "realtime" && (
              <RealtimeWebcamCapture onPrediction={handleRealtimePrediction} />
            )}
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {mode === "realtime" && realtimePrediction && (
              <TranslationResult
                result={{
                  predicted_gloss: realtimePrediction.gloss,
                  confidence: realtimePrediction.confidence,
                  top_5_predictions: [{ gloss: realtimePrediction.gloss, confidence: realtimePrediction.confidence }],
                  processing_time_ms: 0,
                  landmark_extraction_time_ms: 0,
                  inference_time_ms: 0,
                }}
              />
            )}

            {mode !== "realtime" && result && (
              <TranslationResult result={result} />
            )}

            {mode !== "realtime" && !result && !isProcessing && (
              <div className="p-8 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 text-center">
                <svg
                  className="w-16 h-16 mx-auto text-gray-300 dark:text-gray-600 mb-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z"
                  />
                </svg>
                <p className="text-gray-500 dark:text-gray-400">
                  {mode === "upload"
                    ? "Upload a video file to get started"
                    : "Record a video to get started"}
                </p>
              </div>
            )}

            {isProcessing && (
              <div className="p-8 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 text-center">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mb-4"></div>
                <p className="text-gray-500 dark:text-gray-400">
                  Processing video...
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

