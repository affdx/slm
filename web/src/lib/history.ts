/**
 * Translation History Storage Utilities
 *
 * Uses IndexedDB for video blob storage (localStorage has size limits)
 * and localStorage for metadata.
 */

export interface HistoryItem {
  id: number;
  timestamp: string;
  source: string;
  prediction: string;
  confidence: number;
  top5: Array<{ gloss: string; confidence: number }>;
  hasVideo: boolean;
  videoType?: string;
}

const DB_NAME = "msl-translator";
const DB_VERSION = 1;
const STORE_NAME = "videos";
const MAX_HISTORY_ITEMS = 20; // Limit to save storage space

/**
 * Open IndexedDB database
 */
function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
  });
}

/**
 * Save video blob to IndexedDB
 */
export async function saveVideoBlob(id: number, blob: Blob): Promise<void> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, "readwrite");
    const store = transaction.objectStore(STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.put({ id, blob, type: blob.type });
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error("[History] Failed to save video:", error);
  }
}

/**
 * Get video blob from IndexedDB
 */
export async function getVideoBlob(id: number): Promise<Blob | null> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, "readonly");
    const store = transaction.objectStore(STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.get(id);
      request.onsuccess = () => {
        const result = request.result;
        resolve(result ? result.blob : null);
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error("[History] Failed to get video:", error);
    return null;
  }
}

/**
 * Delete video blob from IndexedDB
 */
export async function deleteVideoBlob(id: number): Promise<void> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, "readwrite");
    const store = transaction.objectStore(STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.delete(id);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error("[History] Failed to delete video:", error);
  }
}

/**
 * Clear all videos from IndexedDB
 */
export async function clearAllVideos(): Promise<void> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, "readwrite");
    const store = transaction.objectStore(STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.clear();
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error("[History] Failed to clear videos:", error);
  }
}

/**
 * Get history metadata from localStorage
 */
export function getHistoryMetadata(): HistoryItem[] {
  try {
    const stored = localStorage.getItem("translationHistory");
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

/**
 * Save history metadata to localStorage
 */
export function saveHistoryMetadata(history: HistoryItem[]): void {
  try {
    localStorage.setItem("translationHistory", JSON.stringify(history));
  } catch (error) {
    console.error("[History] Failed to save metadata:", error);
  }
}

/**
 * Add a new history item with video
 */
export async function addHistoryItem(
  prediction: {
    predicted_gloss: string;
    confidence: number;
    top_5_predictions: Array<{ gloss: string; confidence: number }>;
  },
  source: string,
  videoBlob?: Blob
): Promise<void> {
  const id = Date.now();

  // Save video if provided
  if (videoBlob) {
    await saveVideoBlob(id, videoBlob);
  }

  // Get existing history
  const history = getHistoryMetadata();

  // Add new item
  history.unshift({
    id,
    timestamp: new Date().toISOString(),
    source,
    prediction: prediction.predicted_gloss,
    confidence: prediction.confidence,
    top5: prediction.top_5_predictions,
    hasVideo: !!videoBlob,
    videoType: videoBlob?.type,
  });

  // Remove old items and their videos
  if (history.length > MAX_HISTORY_ITEMS) {
    const removed = history.splice(MAX_HISTORY_ITEMS);
    for (const item of removed) {
      if (item.hasVideo) {
        await deleteVideoBlob(item.id);
      }
    }
  }

  // Save metadata
  saveHistoryMetadata(history);
}

/**
 * Delete a history item
 */
export async function deleteHistoryItem(id: number): Promise<HistoryItem[]> {
  const history = getHistoryMetadata();
  const item = history.find((h) => h.id === id);

  if (item?.hasVideo) {
    await deleteVideoBlob(id);
  }

  const updated = history.filter((h) => h.id !== id);
  saveHistoryMetadata(updated);

  return updated;
}

/**
 * Clear all history
 */
export async function clearAllHistory(): Promise<void> {
  await clearAllVideos();
  localStorage.removeItem("translationHistory");
}

/**
 * Create object URL for video playback
 */
export async function getVideoUrl(id: number): Promise<string | null> {
  const blob = await getVideoBlob(id);
  if (blob) {
    return URL.createObjectURL(blob);
  }
  return null;
}

/**
 * Generate thumbnail from video blob
 */
export async function generateThumbnail(
  blob: Blob,
  time: number = 0.5
): Promise<string | null> {
  return new Promise((resolve) => {
    try {
      const video = document.createElement("video");
      video.playsInline = true;
      video.muted = true;
      video.src = URL.createObjectURL(blob);

      video.onloadedmetadata = () => {
        video.currentTime = Math.min(time, video.duration * 0.5);
      };

      video.onseeked = () => {
        const canvas = document.createElement("canvas");
        canvas.width = 160;
        canvas.height = 120;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
          URL.revokeObjectURL(video.src);
          resolve(dataUrl);
        } else {
          resolve(null);
        }
      };

      video.onerror = () => {
        URL.revokeObjectURL(video.src);
        resolve(null);
      };
    } catch {
      resolve(null);
    }
  });
}
