"use client";

import { useState, useRef, useCallback, useEffect } from "react";

interface WebcamCaptureProps {
  onCapture: (video: Blob) => void;
  onRecordStart?: () => void;
  isProcessing: boolean;
  isModelReady?: boolean;
}

export function WebcamCapture({ onCapture, onRecordStart, isProcessing, isModelReady = true }: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [recordingDuration, setRecordingDuration] = useState(0);

  // Stop webcam stream
  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  }, []);

  // Start webcam stream
  const startStream = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
      }
    } catch (err) {
      if (err instanceof Error) {
        if (err.name === "NotAllowedError") {
          setError("Camera access denied. Please allow camera access to use this feature.");
        } else if (err.name === "NotFoundError") {
          setError("No camera found. Please connect a camera and try again.");
        } else {
          setError(`Failed to access camera: ${err.message}`);
        }
      }
    }
  }, []);

  /**
   * Get supported MIME type for MediaRecorder
   * Firefox doesn't support VP9, so we fall back to VP8 or default
   */
  const getSupportedMimeType = useCallback((): string | undefined => {
    const mimeTypes = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
      "video/mp4",
    ];
    
    for (const mimeType of mimeTypes) {
      if (MediaRecorder.isTypeSupported(mimeType)) {
        console.log(`[WebcamCapture] Using MIME type: ${mimeType}`);
        return mimeType;
      }
    }
    
    console.log("[WebcamCapture] No specific MIME type supported, using default");
    return undefined; // Let browser choose default
  }, []);

  // Start recording with countdown
  const startRecording = useCallback(() => {
    if (!videoRef.current?.srcObject) return;

    // Clear previous results when starting new recording
    onRecordStart?.();

    setCountdown(3);

    const countdownInterval = setInterval(() => {
      setCountdown((prev) => {
        if (prev === null || prev <= 1) {
          clearInterval(countdownInterval);
          return null;
        }
        return prev - 1;
      });
    }, 1000);

    setTimeout(() => {
      const stream = videoRef.current!.srcObject as MediaStream;
      
      // Get supported MIME type for cross-browser compatibility
      const mimeType = getSupportedMimeType();
      const options: MediaRecorderOptions = mimeType ? { mimeType } : {};
      
      const mediaRecorder = new MediaRecorder(stream, options);

      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        // Use the actual MIME type from recorder for the blob
        const blobType = mediaRecorder.mimeType || "video/webm";
        const blob = new Blob(chunksRef.current, { type: blobType });
        setRecordedBlob(blob);
        if (previewUrl) {
          URL.revokeObjectURL(previewUrl);
        }
        setPreviewUrl(URL.createObjectURL(blob));
        setIsRecording(false);
        setRecordingDuration(0);
        
        // Auto-submit for inference
        onCapture(blob);
      };

      mediaRecorder.start();
      setIsRecording(true);

      // Track recording duration
      const durationInterval = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);

      // Auto-stop after 5 seconds
      setTimeout(() => {
        if (mediaRecorderRef.current?.state === "recording") {
          mediaRecorderRef.current.stop();
          clearInterval(durationInterval);
        }
      }, 5000);

      // Store interval reference for manual stop
      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };
    }, 3000);
  }, [previewUrl, onRecordStart, getSupportedMimeType]);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  // Reset to recording mode
  const resetRecording = useCallback(() => {
    setRecordedBlob(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  }, [previewUrl]);

  // Auto-start webcam when component mounts, stop when unmounting
  useEffect(() => {
    startStream();

    // Cleanup: stop stream when component unmounts (user switches away from webcam tab)
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run only on mount/unmount
  
  // Cleanup preview URL when it changes
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  return (
    <div className="space-y-4">
      {/* Error Display */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-red-700 dark:text-red-300 text-sm">{error}</p>
        </div>
      )}

      {/* Video Display */}
      <div className="relative rounded-xl overflow-hidden bg-gray-900 aspect-video" role="region" aria-label="Video preview area">
        {!isStreaming && !recordedBlob && !error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white" aria-live="polite">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-gray-400 border-t-white mb-4" aria-hidden="true"></div>
            <p className="text-gray-400">Starting camera...</p>
          </div>
        )}

        {/* Live video feed */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover ${recordedBlob ? "hidden" : ""}`}
          aria-label="Live webcam feed"
        />

        {/* Preview recorded video */}
        {previewUrl && (
          <video src={previewUrl} controls className="w-full h-full object-contain" aria-label="Recorded video preview" />
        )}

        {/* Countdown Overlay */}
        {countdown !== null && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center" role="status" aria-live="assertive">
            <span className="text-8xl font-bold text-white animate-pulse">{countdown}</span>
            <span className="sr-only">Recording starts in {countdown} seconds</span>
          </div>
        )}

        {/* Recording Indicator */}
        {isRecording && (
          <div className="absolute top-4 left-4 flex items-center space-x-2 bg-red-600 text-white px-3 py-1 rounded-full" role="status" aria-live="polite">
            <span className="w-3 h-3 bg-white rounded-full animate-pulse" aria-hidden="true"></span>
            <span className="text-sm font-medium">REC {recordingDuration}s / 5s</span>
            <span className="sr-only">Recording in progress: {recordingDuration} of 5 seconds</span>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-3 justify-center" role="group" aria-label="Recording controls">
        {isStreaming && !isRecording && !recordedBlob && countdown === null && (
          <button
            onClick={startRecording}
            aria-label="Start recording with 3 second countdown"
            className="px-6 py-3 bg-red-600 text-white font-medium rounded-lg hover:bg-red-700 transition-colors flex items-center space-x-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-2"
          >
            <span className="w-3 h-3 bg-white rounded-full" aria-hidden="true"></span>
            <span>Record (3s countdown)</span>
          </button>
        )}

        {isRecording && (
          <button
            onClick={stopRecording}
            aria-label="Stop recording"
            className="px-6 py-3 bg-gray-800 text-white font-medium rounded-lg hover:bg-gray-900 transition-colors flex items-center space-x-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-500 focus-visible:ring-offset-2"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <rect x="6" y="6" width="12" height="12" rx="1" />
            </svg>
            <span>Stop Recording</span>
          </button>
        )}

        {recordedBlob && (
          <>
            {isProcessing ? (
              <div className="px-6 py-3 bg-primary-600/80 text-white font-medium rounded-lg flex items-center space-x-2" role="status" aria-live="polite">
                <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent" aria-hidden="true"></div>
                <span>Processing...</span>
              </div>
            ) : (
              <button
                onClick={resetRecording}
                aria-label="Record a new video"
                className="px-6 py-3 bg-gray-600 text-white font-medium rounded-lg hover:bg-gray-700 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-500 focus-visible:ring-offset-2"
              >
                Record Again
              </button>
            )}
          </>
        )}
      </div>

      {/* Instructions */}
      <div className="text-center text-sm text-gray-500 dark:text-gray-400 space-y-1">
        <p>Position yourself in the frame with good lighting</p>
        <p>Recording auto-translates after 5 seconds</p>
      </div>
    </div>
  );
}
