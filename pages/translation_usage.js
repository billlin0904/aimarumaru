"use strict";

(function attachTranslationUsage(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.TranslationUsage = api;
}(typeof globalThis !== "undefined" ? globalThis : this, () => {
  function emptyTranslationUsageTotals() {
    return {
      countedInputTokens: 0,
      promptTokens: 0,
      cachedInputTokens: 0,
      outputTokens: 0,
      thoughtsTokens: 0,
      totalTokens: 0,
      estimatedCostUsd: 0,
      estimatedCostTwd: 0,
    };
  }

  function addTranslationUsage(totals, usage) {
    if (!usage || typeof usage !== "object") return totals;
    const fields = {
      countedInputTokens: "counted_input_tokens",
      promptTokens: "prompt_tokens",
      cachedInputTokens: "cached_input_tokens",
      outputTokens: "output_tokens",
      thoughtsTokens: "thoughts_tokens",
      totalTokens: "total_tokens",
      estimatedCostUsd: "estimated_total_cost_usd",
      estimatedCostTwd: "estimated_total_cost_twd",
    };
    for (const [target, source] of Object.entries(fields)) {
      const value = Number(usage[source]);
      if (Number.isFinite(value)) totals[target] += value;
    }
    return totals;
  }

  return { addTranslationUsage, emptyTranslationUsageTotals };
}));
