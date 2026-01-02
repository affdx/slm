import Link from "next/link";

export function Footer() {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="bg-white/50 dark:bg-slate-900/50 border-t border-gray-200 dark:border-gray-800 backdrop-blur-sm">
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-center md:text-left">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-gray-900 dark:text-white">MSL Translator</span>
            <span className="text-gray-300 dark:text-gray-700 hidden md:inline">|</span>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Breaking communication barriers.
            </p>
          </div>
          
          <div className="flex items-center gap-6">
            <Link
              href="/about"
              className="text-sm text-gray-500 hover:text-primary-600 dark:text-gray-400 dark:hover:text-primary-400 transition-colors"
            >
              About
            </Link>
            <Link
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer" 
              className="text-sm text-gray-500 hover:text-primary-600 dark:text-gray-400 dark:hover:text-primary-400 transition-colors"
            >
              GitHub
            </Link>
            <span className="text-sm text-gray-400 dark:text-gray-600">© {currentYear}</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
