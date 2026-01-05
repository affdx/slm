"use client";

import { useState, useEffect, useMemo } from "react";
import { getGlosses } from "@/lib/inference";
import { formatGlossName } from "@/hooks/useSignLanguageInference";

export default function DictionaryPage() {
  const [glosses, setGlosses] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");

  useEffect(() => {
    async function loadGlosses() {
      try {
        const data = await getGlosses();
        setGlosses(data);
      } catch (err) {
        setError("Failed to load glosses from local data.");
      } finally {
        setLoading(false);
      }
    }
    loadGlosses();
  }, []);

  const categories = useMemo(() => {
    const cats: Record<string, string[]> = {
      greetings: ["assalamualaikum", "hi", "apa_khabar", "baik", "baik_2"],
      family: [
        "abang", "kakak", "ayah", "bapa", "emak", "anak_lelaki", "anak_perempuan",
        "bapa_saudara", "emak_saudara", "saudara", "keluarga"
      ],
      questions: ["apa", "bagaimana", "bila", "mana", "siapa", "berapa"],
      actions: [
        "ambil", "baca", "bawa", "beli", "beli_2", "buat", "buang", "dapat",
        "jumpa", "main", "makan", "minum", "pergi", "pergi_2", "pinjam",
        "pukul", "tidur", "berjalan", "berlari", "curi", "lupa"
      ],
      descriptions: [
        "baik", "jahat", "pandai", "pandai_2", "perlahan", "perlahan_2",
        "panas", "panas_2", "sejuk", "marah", "suka", "kesakitan"
      ],
      places: ["sekolah", "tandas", "arah"],
      transport: ["bas", "kereta", "teksi"],
      objects: ["bola", "payung", "pen", "pensil", "jam"],
      food: ["nasi", "nasi_lemak", "teh_tarik", "lemak"],
      weather: ["hujan", "ribut"],
      other: [],
    };

    const categorized = new Set(Object.values(cats).flat());
    glosses.forEach((gloss) => {
      if (!categorized.has(gloss)) {
        cats.other.push(gloss);
      }
    });

    return cats;
  }, [glosses]);

  const filteredGlosses = useMemo(() => {
    let filtered = glosses;

    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(
        (gloss) =>
          gloss.toLowerCase().includes(term) ||
          formatGlossName(gloss).toLowerCase().includes(term)
      );
    }

    if (selectedCategory !== "all") {
      const categoryGlosses = categories[selectedCategory] || [];
      filtered = filtered.filter((gloss) => categoryGlosses.includes(gloss));
    }

    return filtered;
  }, [glosses, searchTerm, selectedCategory, categories]);

  if (loading) {
    return (
      <div className="min-h-screen pt-24 flex items-center justify-center">
        <div className="flex flex-col items-center">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-200 border-t-primary-600 mb-4"></div>
          <p className="text-gray-500">Loading dictionary...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen pt-24 container mx-auto px-4">
        <div className="max-w-md mx-auto text-center p-8 bg-red-50 dark:bg-red-900/10 rounded-2xl border border-red-100 dark:border-red-800">
          <p className="text-red-600 dark:text-red-400">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-slate-950 pt-32 pb-20">
      <div className="container mx-auto px-4">
        <div className="max-w-3xl mx-auto text-center mb-16 relative z-0">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Sign Language Dictionary
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Explore our collection of {glosses.length} supported signs and gestures.
          </p>
        </div>

        <div className="sticky top-16 z-40 bg-gray-50 dark:bg-slate-950 py-4 mb-8 -mx-4 px-4 border-b border-gray-200 dark:border-gray-800 shadow-sm dark:shadow-none">
          <div className="max-w-6xl mx-auto space-y-6">
            <div className="relative max-w-2xl mx-auto">
              <svg
                className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                placeholder="Search for a sign..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                aria-label="Search for a sign in the dictionary"
                className="w-full pl-12 pr-4 py-4 rounded-2xl border-none bg-white dark:bg-gray-800 shadow-lg shadow-gray-200/50 dark:shadow-none ring-1 ring-gray-200 dark:ring-gray-700 focus:ring-2 focus:ring-primary-500 transition-all text-lg focus-visible:outline-none"
              />
            </div>

            <div className="flex flex-wrap justify-center gap-2" role="group" aria-label="Filter by category">
              <button
                onClick={() => setSelectedCategory("all")}
                aria-pressed={selectedCategory === "all"}
                className={`px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2 ${
                  selectedCategory === "all"
                    ? "bg-primary-600 text-white shadow-lg shadow-primary-500/25"
                    : "bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600"
                }`}
              >
                All Signs
              </button>
              {Object.entries(categories).map(([category, items]) => {
                if (items.length === 0) return null;
                return (
                  <button
                    key={category}
                    onClick={() => setSelectedCategory(category)}
                    aria-pressed={selectedCategory === category}
                    className={`px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-200 capitalize focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2 ${
                      selectedCategory === category
                        ? "bg-primary-600 text-white shadow-lg shadow-primary-500/25"
                        : "bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600"
                    }`}
                  >
                    {category}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto relative z-0">
          {filteredGlosses.length === 0 ? (
            <div className="text-center py-20">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gray-100 dark:bg-gray-800 mb-4">
                <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="text-gray-500 dark:text-gray-400 text-lg">No signs found matching your criteria.</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
              {filteredGlosses.map((gloss, index) => (
                <div
                  key={gloss}
                  className="group bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-100 dark:border-gray-800 hover:border-primary-200 dark:hover:border-primary-800 hover:shadow-lg hover:shadow-primary-500/5 dark:hover:shadow-none transition-all duration-300"
                >
                  <div className="text-xs font-medium text-gray-400 dark:text-gray-500 mb-2 font-mono">
                    #{String(index + 1).padStart(3, '0')}
                  </div>
                  <h3 className="font-semibold text-gray-900 dark:text-white text-lg group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                    {formatGlossName(gloss)}
                  </h3>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
