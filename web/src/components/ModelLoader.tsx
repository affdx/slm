"use client";

interface ModelLoaderProps {
  isLoading: boolean;
  progress: string;
  error: string | null;
}

export function ModelLoader({ isLoading, progress, error }: ModelLoaderProps) {
  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-8">
        <div className="flex items-center">
          <svg
            className="w-5 h-5 text-red-500 mr-3"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
          <div>
            <p className="font-medium text-red-800 dark:text-red-200">
              Failed to load AI models
            </p>
            <p className="text-sm text-red-600 dark:text-red-300 mt-1">
              {error}
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (!isLoading) {
    return null;
  }

  return (
    <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-8">
      <div className="flex items-center">
        <div className="mr-4">
          <svg
            className="animate-spin h-6 w-6 text-blue-600 dark:text-blue-400"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            ></circle>
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            ></path>
          </svg>
        </div>
        <div>
          <p className="font-medium text-blue-800 dark:text-blue-200">
            Loading AI Models
          </p>
          <p className="text-sm text-blue-600 dark:text-blue-300 mt-1">
            {progress || "Initializing..."}
          </p>
        </div>
      </div>
      {/* Progress bar */}
      <div className="mt-3 h-1.5 w-full bg-blue-200 dark:bg-blue-800 rounded-full overflow-hidden">
        <div className="h-full bg-blue-600 dark:bg-blue-400 rounded-full animate-pulse w-2/3"></div>
      </div>
    </div>
  );
}
