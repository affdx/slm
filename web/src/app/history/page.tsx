"use client";

import { useState, useEffect } from "react";
import { formatGlossName, formatConfidence } from "@/lib/api";

interface HistoryItem {
  id: number;
  timestamp: string;
  source: string;
  prediction: string;
  confidence: number;
  top5: Array<{ gloss: string; confidence: number }>;
}

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem("translationHistory");
    if (stored) {
      setHistory(JSON.parse(stored));
    }
  }, []);

  const clearHistory = () => {
    if (confirm("Are you sure you want to clear all history?")) {
      localStorage.removeItem("translationHistory");
      setHistory([]);
      setSelectedItem(null);
    }
  };

  const deleteItem = (id: number) => {
    const updated = history.filter((item) => item.id !== id);
    localStorage.setItem("translationHistory", JSON.stringify(updated));
    setHistory(updated);
    if (selectedItem?.id === id) {
      setSelectedItem(null);
    }
  };

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString("en-MY", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Translation History
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            {history.length} translation{history.length !== 1 ? "s" : ""} saved locally
          </p>
        </div>
        {history.length > 0 && (
          <button
            onClick={clearHistory}
            className="mt-4 sm:mt-0 px-4 py-2 text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
          >
            Clear All History
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="text-center py-16">
          <svg
            className="w-16 h-16 text-gray-400 mx-auto mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <h2 className="text-xl font-medium text-gray-900 dark:text-white mb-2">
            No translations yet
          </h2>
          <p className="text-gray-500 dark:text-gray-400">
            Your translation history will appear here after you translate some signs.
          </p>
        </div>
      ) : (
        <div className="grid lg:grid-cols-3 gap-6">
          {/* History List */}
          <div className="lg:col-span-2 space-y-3">
            {history.map((item) => (
              <div
                key={item.id}
                onClick={() => setSelectedItem(item)}
                className={`bg-white dark:bg-gray-800 rounded-lg p-4 border transition-all cursor-pointer ${
                  selectedItem?.id === item.id
                    ? "border-primary-500 ring-2 ring-primary-200 dark:ring-primary-900"
                    : "border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600"
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        {formatGlossName(item.prediction)}
                      </h3>
                      <span
                        className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                          item.confidence >= 0.8
                            ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                            : item.confidence >= 0.5
                              ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400"
                              : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
                        }`}
                      >
                        {formatConfidence(item.confidence)}
                      </span>
                    </div>
                    <div className="flex items-center space-x-4 mt-2 text-sm text-gray-500 dark:text-gray-400">
                      <span>{formatDate(item.timestamp)}</span>
                      <span className="truncate max-w-[200px]">{item.source}</span>
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteItem(item.id);
                    }}
                    className="p-2 text-gray-400 hover:text-red-500 transition-colors"
                    aria-label="Delete"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>

          {/* Detail Panel */}
          <div className="lg:col-span-1">
            {selectedItem ? (
              <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 sticky top-4">
                <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
                  Translation Details
                </h3>
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                  {formatGlossName(selectedItem.prediction)}
                </h2>

                <div className="space-y-4">
                  <div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">Confidence</span>
                    <div className="flex items-center space-x-2 mt-1">
                      <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${
                            selectedItem.confidence >= 0.8
                              ? "bg-green-500"
                              : selectedItem.confidence >= 0.5
                                ? "bg-yellow-500"
                                : "bg-red-500"
                          }`}
                          style={{ width: `${selectedItem.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {formatConfidence(selectedItem.confidence)}
                      </span>
                    </div>
                  </div>

                  <div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">Top 5 Predictions</span>
                    <div className="mt-2 space-y-2">
                      {selectedItem.top5.map((pred, idx) => (
                        <div key={pred.gloss} className="flex items-center justify-between text-sm">
                          <span className="text-gray-700 dark:text-gray-300">
                            {idx + 1}. {formatGlossName(pred.gloss)}
                          </span>
                          <span className="text-gray-500 dark:text-gray-400">
                            {formatConfidence(pred.confidence)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                    <div className="text-sm">
                      <span className="text-gray-500 dark:text-gray-400">Source:</span>
                      <p className="text-gray-900 dark:text-white mt-1 break-all">
                        {selectedItem.source}
                      </p>
                    </div>
                    <div className="text-sm mt-3">
                      <span className="text-gray-500 dark:text-gray-400">Date:</span>
                      <p className="text-gray-900 dark:text-white mt-1">
                        {formatDate(selectedItem.timestamp)}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-dashed border-gray-300 dark:border-gray-600 p-6 text-center">
                <p className="text-gray-500 dark:text-gray-400">
                  Select a translation to view details
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
