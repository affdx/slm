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

  // Categorize glosses
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

    // Find uncategorized glosses
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

    // Filter by search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(
        (gloss) =>
          gloss.toLowerCase().includes(term) ||
          formatGlossName(gloss).toLowerCase().includes(term)
      );
    }

    // Filter by category
    if (selectedCategory !== "all") {
      const categoryGlosses = categories[selectedCategory] || [];
      filtered = filtered.filter((gloss) => categoryGlosses.includes(gloss));
    }

    return filtered;
  }, [glosses, searchTerm, selectedCategory, categories]);

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-12">
        <div className="flex flex-col items-center justify-center min-h-[400px]">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-200 border-t-primary-600 mb-4"></div>
          <p className="text-gray-600 dark:text-gray-300">Loading dictionary...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-md mx-auto text-center">
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
            <svg
              className="w-12 h-12 text-red-500 mx-auto mb-4"
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
            <p className="text-red-700 dark:text-red-300">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Sign Language Dictionary
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Browse all {glosses.length} Malaysian Sign Language glosses
        </p>
      </div>

      {/* Filters */}
      <div className="max-w-4xl mx-auto mb-8 space-y-4">
        {/* Search */}
        <div className="relative">
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          <input
            type="text"
            placeholder="Search glosses..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setSelectedCategory("all")}
            className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
              selectedCategory === "all"
                ? "bg-primary-600 text-white"
                : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
            }`}
          >
            All ({glosses.length})
          </button>
          {Object.entries(categories).map(([category, items]) => {
            if (items.length === 0) return null;
            return (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-colors capitalize ${
                  selectedCategory === category
                    ? "bg-primary-600 text-white"
                    : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
                }`}
              >
                {category} ({items.length})
              </button>
            );
          })}
        </div>
      </div>

      {/* Glosses Grid */}
      <div className="max-w-6xl mx-auto">
        {filteredGlosses.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-gray-500 dark:text-gray-400">No glosses found matching your search.</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {filteredGlosses.map((gloss, index) => (
              <div
                key={gloss}
                className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700 hover:border-primary-500 hover:shadow-md transition-all cursor-default"
              >
                <p className="font-medium text-gray-900 dark:text-white text-center">
                  {formatGlossName(gloss)}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400 text-center mt-1">
                  #{index + 1}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
