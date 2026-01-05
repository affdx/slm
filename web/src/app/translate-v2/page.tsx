"use client";

import { useState, useCallback } from "react";
import { RealtimeWebcamCaptureV2 } from "@/components/RealtimeWebcamCaptureV2";
import { ModelLoader } from "@/components/ModelLoader";
import {
  useSignLanguageInference,
} from "@/hooks/useSignLanguageInference";
import { DetectedSign } from "@/hooks/useRealtimeInferenceV2";

export default function TranslateV2Page() {
  const [realtimePrediction, setRealtimePrediction] = useState<DetectedSign | null>(null);

  const { 
    isReady, 
    isLoading, 
    loadingProgress, 
    error: modelError, 
    currentModel,
    availableModels,
    switchModel,
  } = useSignLanguageInference();

  const handleRealtimePrediction = useCallback((sign: DetectedSign) => {
    setRealtimePrediction(sign);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50/50 dark:bg-slate-900/50 pt-24 pb-12">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 gap-4">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white tracking-tight">
                Live Detect V2
              </h1>
              <span className="px-2 py-0.5 bg-purple-600 text-white text-xs font-bold rounded">
                EXPERIMENTAL
              </span>
            </div>
            <p className="text-gray-500 dark:text-gray-400 mt-1">
              Python-style sliding window with continuous inference (~10 pred/sec)
            </p>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-purple-50 dark:bg-purple-900/20 border border-purple-100 dark:border-purple-800 text-xs font-medium text-purple-700 dark:text-purple-400">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500"></span>
              </span>
              Sliding Window Mode
            </div>
             
            {isReady && (
              <select
                value={currentModel}
                onChange={(e) => switchModel(e.target.value as typeof currentModel)}
                disabled={isLoading}
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

        {/* V1 vs V2 Comparison Info */}
        <div className="mb-6 p-4 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-xl border border-purple-100 dark:border-purple-800/50">
          <h3 className="text-sm font-semibold text-purple-800 dark:text-purple-300 mb-2">
            What&apos;s different in V2?
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div className="space-y-1">
              <p className="font-medium text-gray-700 dark:text-gray-300">V1 (Original):</p>
              <ul className="text-gray-600 dark:text-gray-400 space-y-0.5">
                <li>• Collects 30 frames, clears buffer, repeats</li>
                <li>• ~1 prediction per second</li>
                <li>• Sign must fit within exact 30-frame window</li>
              </ul>
            </div>
            <div className="space-y-1">
              <p className="font-medium text-purple-700 dark:text-purple-300">V2 (Python-style):</p>
              <ul className="text-purple-600 dark:text-purple-400 space-y-0.5">
                <li>• Sliding window: shifts by 1 frame, overlaps 29/30</li>
                <li>• ~10 predictions per second</li>
                <li>• Catches signs mid-gesture, better temporal coverage</li>
              </ul>
            </div>
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
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 overflow-hidden shadow-sm min-h-[400px] flex flex-col relative">
              <div className="flex-1 p-2">
                <RealtimeWebcamCaptureV2
                  key="realtime-v2"
                  onPrediction={handleRealtimePrediction}
                />
              </div>
            </div>
          </div>

          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm h-full max-h-[600px] flex flex-col">
              <h3 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-6">
                Live Detection V2
              </h3>
              
              {realtimePrediction ? (
                <div className="flex-1 flex flex-col items-center justify-center text-center space-y-6">
                  <div className="w-32 h-32 rounded-full bg-purple-50 dark:bg-purple-900/20 flex items-center justify-center mb-2 animate-pulse">
                    <svg className="w-16 h-16 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  
                  <div>
                    <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
                      {realtimePrediction.gloss}
                    </h2>
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 font-medium">
                      {(realtimePrediction.confidence * 100).toFixed(1)}% Confidence
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-center text-gray-400 space-y-4">
                  <div className="w-20 h-20 rounded-full bg-gray-100 dark:bg-gray-700/50 flex items-center justify-center">
                    <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <p>Waiting for sign input...</p>
                  <p className="text-xs text-gray-500">V2: Continuous sliding window</p>
                </div>
              )}
            </div>

            {/* Tips */}
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
          </div>
        </div>
      </div>
    </div>
  );
}
