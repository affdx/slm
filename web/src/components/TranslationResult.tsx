"use client";

import { PredictionResult } from "@/hooks/useSignLanguageInference";

interface TranslationResultProps {
  result: PredictionResult;
}

function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`;
}

function formatGlossName(gloss: string): string {
  return gloss
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function TranslationResult({ result }: TranslationResultProps) {
  const confidenceColor =
    result.confidence >= 0.8
      ? "text-green-600 dark:text-green-400"
      : result.confidence >= 0.5
        ? "text-yellow-600 dark:text-yellow-400"
        : "text-red-600 dark:text-red-400";

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
      {/* Main Result */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <div className="text-center">
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Translation Result</p>
          <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            {formatGlossName(result.predicted_gloss)}
          </h2>
          <div className="flex items-center justify-center space-x-2">
            <span className="text-sm text-gray-500 dark:text-gray-400">Confidence:</span>
            <span className={`text-lg font-semibold ${confidenceColor}`}>
              {formatConfidence(result.confidence)}
            </span>
          </div>
        </div>

        {/* Confidence Bar */}
        <div className="mt-4">
          <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-500 ${
                result.confidence >= 0.8
                  ? "bg-green-500"
                  : result.confidence >= 0.5
                    ? "bg-yellow-500"
                    : "bg-red-500"
              }`}
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Top 5 Predictions */}
      <div className="p-6">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
          Top 5 Predictions
        </h3>
        <div className="space-y-3">
          {result.top_5_predictions.map((pred, index) => (
            <div key={pred.gloss} className="flex items-center space-x-3">
              <span className="w-6 h-6 flex items-center justify-center text-xs font-medium text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 rounded-full">
                {index + 1}
              </span>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <span
                    className={`font-medium ${
                      index === 0
                        ? "text-primary-600 dark:text-primary-400"
                        : "text-gray-700 dark:text-gray-300"
                    }`}
                  >
                    {formatGlossName(pred.gloss)}
                  </span>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {formatConfidence(pred.confidence)}
                  </span>
                </div>
                <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      index === 0 ? "bg-primary-500" : "bg-gray-400 dark:bg-gray-500"
                    }`}
                    style={{ width: `${pred.confidence * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Processing Time Breakdown */}
      <div className="px-6 py-3 bg-gray-50 dark:bg-gray-700/50">
        <div className="flex flex-wrap items-center justify-center gap-4 text-xs text-gray-500 dark:text-gray-400">
          <span>
            Total: <strong>{result.processing_time_ms.toFixed(0)}ms</strong>
          </span>
          {result.landmark_extraction_time_ms !== undefined && (
            <span>
              Landmarks: <strong>{result.landmark_extraction_time_ms.toFixed(0)}ms</strong>
            </span>
          )}
          {result.inference_time_ms !== undefined && (
            <span>
              Inference: <strong>{result.inference_time_ms.toFixed(0)}ms</strong>
            </span>
          )}
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">
            Client-side
          </span>
        </div>
      </div>
    </div>
  );
}
