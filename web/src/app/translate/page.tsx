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

type InputMode = "upload" | "webcam" | "realtime";
type UploadSubMode = "quick" | "demo";

export default function TranslatePage() {
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
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white tracking-tight">
              Translate
            </h1>
            <p className="text-gray-500 dark:text-gray-400 mt-1">
              Real-time Malaysian Sign Language translation
            </p>
          </div>
          
          <div className="flex items-center gap-3">
             <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-100 dark:border-emerald-800 text-xs font-medium text-emerald-700 dark:text-emerald-400">
               <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              Client-side Inference
            </div>
            
            {isReady && (
              <select
                value={currentModel}
                onChange={(e) => switchModel(e.target.value as typeof currentModel)}
                disabled={isLoading || isProcessing}
                className="text-xs font-medium px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500/20"
              >
                {availableModels.map(({ type, config }) => (
                  <option key={type} value={type}>
                    {config.name}
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>

        {(isLoading || modelError) && (
          <div className="mb-8">
            <ModelLoader
              isLoading={isLoading}
              progress={loadingProgress}
              error={modelError}
            />
          </div>
        )}

        <div className="grid lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white dark:bg-gray-800 p-1 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 inline-flex w-full sm:w-auto">
              {(["upload", "webcam", "realtime"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => { setMode(m); clearResults(); }}
                  className={`flex-1 sm:flex-none px-6 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                    mode === m
                      ? "bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm"
                      : "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
                  }`}
                >
                  {m === "upload" && "Upload Video"}
                  {m === "webcam" && "Record"}
                  {m === "realtime" && "Live Detect"}
                </button>
              ))}
            </div>

            {mode === "upload" && (
              <div className="flex gap-2">
                {(["quick", "demo"] as const).map((sm) => (
                  <button
                    key={sm}
                    onClick={() => { setUploadSubMode(sm); clearResults(); }}
                    className={`px-4 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                      uploadSubMode === sm
                        ? "bg-primary-50 border-primary-200 text-primary-700 dark:bg-primary-900/20 dark:border-primary-800 dark:text-primary-300"
                        : "bg-transparent border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400"
                    }`}
                  >
                    {sm === "quick" ? "Quick Result" : "Frame-by-Frame Analysis"}
                  </button>
                ))}
              </div>
            )}

            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 overflow-hidden shadow-sm min-h-[400px] flex flex-col relative">
              <div className="flex-1 p-2">
                {mode === "upload" && !selectedVideoBlob && (
                  <VideoUpload
                    onSubmit={handleVideoSubmit}
                    onFileChange={clearResults}
                    isProcessing={isProcessing}
                    isModelReady={isReady}
                  />
                )}
                
                {mode === "upload" && selectedVideoBlob && uploadSubMode === "demo" && (
                  <div className="h-full flex flex-col">
                    <div className="flex justify-between items-center px-4 py-2 border-b border-gray-100 dark:border-gray-700 mb-4">
                      <h3 className="text-sm font-medium text-gray-900 dark:text-white">Analysis View</h3>
                      <button
                        onClick={clearResults}
                        className="text-xs text-primary-600 hover:text-primary-700 font-medium"
                      >
                        Change Video
                      </button>
                    </div>
                    <div className="flex-1">
                      <VideoInferencePlayer
                        videoBlob={selectedVideoBlob}
                        filename={selectedFilename || undefined}
                      />
                    </div>
                  </div>
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

              {isProcessing && (
                <div className="absolute inset-0 bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm z-10 flex flex-col items-center justify-center">
                  <div className="w-16 h-16 border-4 border-primary-100 border-t-primary-500 rounded-full animate-spin mb-4" />
                  <p className="text-lg font-medium text-gray-900 dark:text-white">Processing Video</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Extracting landmarks & analyzing gestures...</p>
                </div>
              )}
            </div>

            {error && (
              <div className="p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-100 dark:border-red-800 flex items-center gap-3 text-red-700 dark:text-red-300">
                <svg className="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {error}
              </div>
            )}
          </div>

          <div className="lg:col-span-1 space-y-6">
            {mode === "realtime" && (
               <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm h-full max-h-[600px] flex flex-col">
                <h3 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-6">
                  Live Detection
                </h3>
                
                {realtimePrediction ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-center space-y-6">
                    <div className="w-32 h-32 rounded-full bg-emerald-50 dark:bg-emerald-900/20 flex items-center justify-center mb-2 animate-pulse">
                      <svg className="w-16 h-16 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    
                    <div>
                      <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
                        {realtimePrediction.gloss}
                      </h2>
                      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 font-medium">
                        {(realtimePrediction.confidence * 100).toFixed(1)}% Confidence
                      </div>
                    </div>
                  </div>
                ) : (
                   <div className="flex-1 flex flex-col items-center justify-center text-center text-gray-400 space-y-4">
                    <div className="w-20 h-20 rounded-full bg-gray-100 dark:bg-gray-700/50 flex items-center justify-center">
                      <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                      </svg>
                    </div>
                    <p>Waiting for sign input...</p>
                  </div>
                )}
              </div>
            )}

            {result && !isProcessing && mode !== "realtime" && (
              <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 overflow-hidden shadow-sm">
                 <div className="p-6 border-b border-gray-100 dark:border-gray-700">
                  <h3 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Translation Result
                  </h3>
                </div>
                <div className="p-6">
                  <TranslationResult result={result} />
                </div>
              </div>
            )}
            
            {!result && !realtimePrediction && (
              <div className="bg-primary-50 dark:bg-primary-900/10 rounded-2xl p-6 border border-primary-100 dark:border-primary-800/30">
                <h3 className="text-primary-800 dark:text-primary-300 font-semibold mb-3">Tips for best results</h3>
                <ul className="space-y-2 text-sm text-primary-700 dark:text-primary-400">
                  <li className="flex items-start gap-2">
                    <span className="mt-1">•</span>
                    Ensure good lighting on your hands and face
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="mt-1">•</span>
                    Keep your upper body visible in the frame
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="mt-1">•</span>
                    Perform signs clearly and at a moderate speed
                  </li>
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
