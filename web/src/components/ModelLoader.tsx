"use client";

interface ModelLoaderProps {
  isLoading: boolean;
  progress: string;
  error: string | null;
}

export function ModelLoader({ isLoading, progress, error }: ModelLoaderProps) {
  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/10 border border-red-100 dark:border-red-900/30 rounded-2xl p-6 mb-8 shadow-sm" role="alert">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 w-10 h-10 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center border border-red-200 dark:border-red-800/50">
            <svg
              className="w-5 h-5 text-red-600 dark:text-red-400"
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
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-red-900 dark:text-red-200">
              Failed to load AI models
            </h3>
            <p className="text-sm text-red-700 dark:text-red-300 mt-1 leading-relaxed">
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
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl p-6 mb-8 shadow-sm relative overflow-hidden" role="status">
      <div className="absolute top-0 right-0 w-64 h-64 bg-primary-500/5 dark:bg-primary-500/10 rounded-full -mr-16 -mt-16 blur-3xl pointer-events-none" />

      <div className="relative flex flex-col sm:flex-row items-center sm:items-start gap-5">
        <div className="flex-shrink-0 relative">
          <div className="w-12 h-12 rounded-xl bg-primary-50 dark:bg-primary-900/20 flex items-center justify-center border border-primary-100 dark:border-primary-800/30">
            <svg
              className="animate-spin h-6 w-6 text-primary-600 dark:text-primary-400"
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
          <span className="absolute -top-1 -right-1 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-primary-500 border-2 border-white dark:border-gray-800"></span>
          </span>
        </div>

        <div className="flex-1 w-full text-center sm:text-left">
          <div className="mb-4">
            <h3 className="font-semibold text-gray-900 dark:text-white text-lg tracking-tight">
              Initializing AI Engine
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Preparing secure client-side translation models.
            </p>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-3 border border-gray-100 dark:border-gray-700/50">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-medium text-primary-700 dark:text-primary-300 uppercase tracking-wider">
                Status
              </span>
              <span className="text-xs font-medium text-gray-600 dark:text-gray-300">
                {progress || "Starting..."}
              </span>
            </div>
            <div className="h-1.5 w-full bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-primary-500 to-primary-400 rounded-full animate-pulse w-3/4"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
