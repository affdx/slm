/**
 * Formatting utilities for sign language translation app
 */

/**
 * Format gloss name for display (e.g., "hello_world" -> "Hello World")
 */
export function formatGlossName(gloss: string): string {
  return gloss
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * Format confidence as percentage (e.g., 0.95 -> "95.0%")
 */
export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`;
}
