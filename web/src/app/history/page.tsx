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
    console.log("[History] Loaded items:", items);
    setHistory(items);
  }, []);

  // Load video when item is selected
  useEffect(() => {
    let currentUrl: string | null = null;
    let isCancelled = false;

    async function loadVideo() {
      // Reset states
      setVideoUrl(null);
      setVideoError(null);

      if (!selectedItem) return;

      // Check if item has video
      if (!selectedItem.hasVideo) {
        console.log("[History] Item has no video:", selectedItem.id);
        return;
      }

      setLoadingVideo(true);
      console.log("[History] Loading video for item:", selectedItem.id);

      try {
        const url = await getVideoUrl(selectedItem.id);
        
        if (isCancelled) {
          if (url) URL.revokeObjectURL(url);
          return;
        }

        if (url) {
          console.log("[History] Video URL created:", url);
          currentUrl = url;
          setVideoUrl(url);
        } else {
          console.log("[History] No video blob found for item:", selectedItem.id);
          setVideoError("Video not found in storage");
        }
      } catch (error) {
        console.error("[History] Error loading video:", error);
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

    // Cleanup: revoke object URL when component unmounts or selection changes
    return () => {
      isCancelled = true;
      if (currentUrl) {
        console.log("[History] Revoking URL:", currentUrl);
        URL.revokeObjectURL(currentUrl);
      }
    };
  }, [selectedItem]);

  const handleClearHistory = useCallback(async () => {
    if (confirm("Are you sure you want to clear all history? This will also delete all saved videos.")) {
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
            onClick={handleClearHistory}
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
                      {/* Video indicator */}
                      {item.hasVideo && (
                        <span className="flex-shrink-0 w-6 h-6 flex items-center justify-center rounded bg-primary-100 dark:bg-primary-900/30">
                          <svg
                            className="w-4 h-4 text-primary-600 dark:text-primary-400"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                            />
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                          </svg>
                        </span>
                      )}
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
                      handleDeleteItem(item.id);
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
              <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden sticky top-4">
                {/* Video Player - Always show section */}
                <div className="aspect-video bg-gray-900 relative">
                  {!selectedItem.hasVideo ? (
                    // No video saved for this item
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400">
                      <svg
                        className="w-12 h-12 mb-2"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={1.5}
                          d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                        />
                      </svg>
                      <p className="text-sm">No video saved</p>
                      <p className="text-xs text-gray-500 mt-1">Older translations don&apos;t have videos</p>
                    </div>
                  ) : loadingVideo ? (
                    // Loading state
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-2 border-white border-t-transparent"></div>
                    </div>
                  ) : videoError ? (
                    // Error state
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-red-400">
                      <svg
                        className="w-12 h-12 mb-2"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={1.5}
                          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                        />
                      </svg>
                      <p className="text-sm">{videoError}</p>
                    </div>
                  ) : videoUrl ? (
                    // Video player
                    <video
                      key={videoUrl}
                      src={videoUrl}
                      controls
                      autoPlay
                      muted
                      className="w-full h-full object-contain"
                      playsInline
                      onError={(e) => {
                        console.error("[History] Video playback error:", e);
                        setVideoError("Failed to play video");
                      }}
                    />
                  ) : (
                    // Fallback - shouldn't reach here
                    <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                      <p>Video not available</p>
                    </div>
                  )}
                </div>

                {/* Details */}
                <div className="p-6">
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

                    {selectedItem.top5 && selectedItem.top5.length > 0 && (
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
                    )}

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
              </div>
            ) : (
              <div className="bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-dashed border-gray-300 dark:border-gray-600 p-6 text-center">
                <svg
                  className="w-12 h-12 text-gray-400 mx-auto mb-3"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
                <p className="text-gray-500 dark:text-gray-400">
                  Select a translation to view details and replay the video
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
