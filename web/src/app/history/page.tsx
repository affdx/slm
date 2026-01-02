"use client";

import { useState, useEffect, useCallback } from "react";
import {
  HistoryItem,
  getHistoryMetadata,
  deleteHistoryItem,
  clearAllHistory,
  getVideoUrl,
} from "@/lib/history";
import { formatGlossName, formatConfidence } from "@/lib/format";

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [loadingVideo, setLoadingVideo] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);

  useEffect(() => {
    const items = getHistoryMetadata();
    setHistory(items);
  }, []);

  useEffect(() => {
    let currentUrl: string | null = null;
    let isCancelled = false;

    async function loadVideo() {
      setVideoUrl(null);
      setVideoError(null);

      if (!selectedItem) return;

      if (!selectedItem.hasVideo) {
        return;
      }

      setLoadingVideo(true);

      try {
        const url = await getVideoUrl(selectedItem.id);
        
        if (isCancelled) {
          if (url) URL.revokeObjectURL(url);
          return;
        }

        if (url) {
          currentUrl = url;
          setVideoUrl(url);
        } else {
          setVideoError("Video not found");
        }
      } catch (error) {
        if (!isCancelled) {
          setVideoError("Failed to load video");
        }
      } finally {
        if (!isCancelled) {
          setLoadingVideo(false);
        }
      }
    }

    loadVideo();

    return () => {
      isCancelled = true;
      if (currentUrl) {
        URL.revokeObjectURL(currentUrl);
      }
    };
  }, [selectedItem]);

  const handleClearHistory = useCallback(async () => {
    if (confirm("Clear all history? This cannot be undone.")) {
      await clearAllHistory();
      setHistory([]);
      setSelectedItem(null);
      setVideoUrl(null);
    }
  }, []);

  const handleDeleteItem = useCallback(async (id: number) => {
    const updated = await deleteHistoryItem(id);
    setHistory(updated);
    if (selectedItem?.id === id) {
      setSelectedItem(null);
      setVideoUrl(null);
    }
  }, [selectedItem]);

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString("en-MY", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="min-h-screen bg-gray-50/50 dark:bg-slate-900/50 pt-32 pb-12">
      <div className="container mx-auto px-4">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between mb-8 gap-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">History</h1>
            <p className="text-gray-500 dark:text-gray-400 mt-1">
              Your recent translation activities
            </p>
          </div>
          {history.length > 0 && (
            <button
              onClick={handleClearHistory}
              className="text-sm font-medium text-red-600 hover:text-red-700 bg-red-50 hover:bg-red-100 dark:bg-red-900/20 dark:hover:bg-red-900/30 px-4 py-2 rounded-lg transition-colors"
            >
              Clear All
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24 bg-white dark:bg-gray-800 rounded-3xl border border-gray-100 dark:border-gray-700">
            <div className="w-20 h-20 bg-gray-50 dark:bg-gray-700 rounded-full flex items-center justify-center mb-6">
              <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-medium text-gray-900 dark:text-white mb-2">No history yet</h3>
            <p className="text-gray-500 dark:text-gray-400">Translations will appear here automatically.</p>
          </div>
        ) : (
          <div className="grid lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-4">
              {history.map((item) => (
                <div
                  key={item.id}
                  onClick={() => setSelectedItem(item)}
                  className={`group bg-white dark:bg-gray-800 rounded-xl p-5 border transition-all cursor-pointer hover:shadow-md ${
                    selectedItem?.id === item.id
                      ? "border-primary-500 ring-1 ring-primary-500 shadow-md"
                      : "border-gray-200 dark:border-gray-700 hover:border-primary-300 dark:hover:border-gray-600"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                        item.hasVideo 
                          ? "bg-primary-50 text-primary-600 dark:bg-primary-900/20 dark:text-primary-400"
                          : "bg-gray-100 text-gray-400 dark:bg-gray-800 dark:text-gray-500"
                      }`}>
                         {item.hasVideo ? (
                          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                          </svg>
                         ) : (
                           <span className="font-bold text-lg">T</span>
                         )}
                      </div>
                      
                      <div>
                        <h3 className="font-bold text-gray-900 dark:text-white text-lg leading-tight">
                          {formatGlossName(item.prediction)}
                        </h3>
                        <div className="flex items-center gap-3 mt-1 text-sm">
                          <span className={`font-medium ${
                            item.confidence >= 0.8 ? "text-emerald-600" :
                            item.confidence >= 0.5 ? "text-amber-600" : "text-red-600"
                          }`}>
                            {formatConfidence(item.confidence)}
                          </span>
                          <span className="text-gray-300 dark:text-gray-600">•</span>
                          <span className="text-gray-500 dark:text-gray-400">{formatDate(item.timestamp)}</span>
                        </div>
                      </div>
                    </div>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteItem(item.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-2 text-gray-400 hover:text-red-500 transition-all"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </div>
              ))}
            </div>

            <div className="lg:col-span-1">
              {selectedItem ? (
                <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 overflow-hidden sticky top-24 shadow-lg">
                  <div className="aspect-video bg-black relative">
                    {!selectedItem.hasVideo ? (
                      <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500">
                        <p>No video recorded</p>
                      </div>
                    ) : loadingVideo ? (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-8 h-8 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                      </div>
                    ) : videoError ? (
                      <div className="absolute inset-0 flex items-center justify-center text-red-400">
                        <p>{videoError}</p>
                      </div>
                    ) : videoUrl ? (
                      <video
                        key={videoUrl}
                        src={videoUrl}
                        controls
                        autoPlay
                        className="w-full h-full object-contain"
                      />
                    ) : null}
                  </div>

                  <div className="p-6">
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
                      {formatGlossName(selectedItem.prediction)}
                    </h2>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-6 font-mono">
                      ID: {selectedItem.id}
                    </p>

                    <div className="space-y-6">
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <span className="text-gray-500 dark:text-gray-400">Confidence Score</span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {formatConfidence(selectedItem.confidence)}
                          </span>
                        </div>
                        <div className="h-2 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              selectedItem.confidence >= 0.8 ? "bg-emerald-500" :
                              selectedItem.confidence >= 0.5 ? "bg-amber-500" : "bg-red-500"
                            }`}
                            style={{ width: `${selectedItem.confidence * 100}%` }}
                          />
                        </div>
                      </div>

                      {selectedItem.top5 && (
                        <div>
                          <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                            Alternative Predictions
                          </h4>
                          <div className="space-y-2">
                            {selectedItem.top5.slice(1).map((pred, idx) => (
                              <div key={pred.gloss} className="flex justify-between text-sm p-2 rounded-lg bg-gray-50 dark:bg-gray-700/50">
                                <span className="text-gray-700 dark:text-gray-300">
                                  {formatGlossName(pred.gloss)}
                                </span>
                                <span className="text-gray-500 dark:text-gray-400 font-mono">
                                  {formatConfidence(pred.confidence)}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="hidden lg:flex flex-col items-center justify-center text-center p-12 bg-gray-50 dark:bg-gray-800/50 rounded-2xl border border-dashed border-gray-300 dark:border-gray-700 text-gray-400 sticky top-24">
                  <p>Select an item to view details</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
