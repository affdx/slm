"use client";

import { useState, useCallback, useRef } from "react";
import { useToast } from "./Toast";

interface VideoUploadProps {
  onSubmit: (video: Blob, filename?: string) => void;
  onFileChange?: () => void;
  isProcessing: boolean;
  isModelReady?: boolean;
}

export function VideoUpload({ onSubmit, onFileChange, isProcessing, isModelReady = true }: VideoUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { showToast } = useToast();

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const processFile = useCallback((file: File) => {
    const validTypes = ["video/mp4", "video/webm", "video/quicktime", "video/x-msvideo"];
    if (!validTypes.includes(file.type)) {
      showToast("Please upload a valid video file (MP4, WebM, MOV, or AVI)", "error");
      return;
    }

    if (file.size > 100 * 1024 * 1024) {
      showToast("File size must be less than 100MB", "error");
      return;
    }

    setSelectedFile(file);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });

    onFileChange?.();
  }, [onFileChange, showToast]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  }, [processFile]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  }, [processFile]);

  const handleSubmit = useCallback(() => {
    if (selectedFile) {
      onSubmit(selectedFile, selectedFile.name);
    }
  }, [selectedFile, onSubmit]);

  const handleClear = useCallback(() => {
    setSelectedFile(null);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    // Clear results when file is removed
    onFileChange?.();
  }, [onFileChange]);

  return (
    <div className="h-full flex flex-col">
      {/* Drop Zone */}
      {!selectedFile && (
        <div
          className={`relative border-2 border-dashed rounded-xl p-8 transition-colors flex-1 flex items-center justify-center ${
            dragActive
              ? "border-primary-500 bg-primary-50 dark:bg-primary-900/20"
              : "border-gray-300 dark:border-gray-600 hover:border-primary-400"
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          role="region"
          aria-label="Video upload drop zone"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            aria-label="Upload video file. Accepted formats: MP4, WebM, MOV, AVI. Maximum size: 100MB"
          />
          <div className="text-center">
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
            <p className="mt-4 text-lg font-medium text-gray-900 dark:text-white">
              Drop your video here
            </p>
            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
              or click to browse from your computer
            </p>
            <p className="mt-2 text-xs text-gray-400 dark:text-gray-500">
              MP4, WebM, MOV, AVI up to 100MB
            </p>
          </div>
        </div>
      )}

      {/* Preview */}
      {selectedFile && previewUrl && (
        <div className="space-y-4 flex-1 flex flex-col">
          <div className="relative rounded-lg overflow-hidden bg-black">
            <video
              src={previewUrl}
              controls
              className="w-full max-h-[400px] object-contain"
            />
          </div>
          <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <div className="flex items-center space-x-3">
              <svg
                className="w-8 h-8 text-primary-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
              <div>
                <p className="font-medium text-gray-900 dark:text-white truncate max-w-[200px] sm:max-w-[400px]">
                  {selectedFile.name}
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
            <button
              onClick={handleClear}
              className="p-2 text-gray-500 hover:text-red-500 transition-colors"
              aria-label="Remove file"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            disabled={isProcessing || !isModelReady}
            aria-label={isProcessing ? "Processing video" : !isModelReady ? "Loading AI model" : "Translate sign language from video"}
            aria-busy={isProcessing}
            className="w-full py-3 px-4 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2"
          >
            {isProcessing ? "Processing..." : !isModelReady ? "Loading model..." : "Translate Sign Language"}
          </button>
        </div>
      )}
    </div>
  );
}
