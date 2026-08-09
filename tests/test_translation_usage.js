"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  addTranslationUsage,
  emptyTranslationUsageTotals,
} = require("../pages/translation_usage.js");

test("accumulates Gemini countTokens, billed tokens, and estimated cost", () => {
  const totals = emptyTranslationUsageTotals();
  addTranslationUsage(totals, {
    counted_input_tokens: 120,
    prompt_tokens: 118,
    cached_input_tokens: 20,
    output_tokens: 30,
    thoughts_tokens: 4,
    total_tokens: 152,
    estimated_total_cost_usd: 0.00042,
    estimated_total_cost_twd: 0.01302,
  });
  addTranslationUsage(totals, {
    counted_input_tokens: 80,
    prompt_tokens: 78,
    cached_input_tokens: 0,
    output_tokens: 20,
    thoughts_tokens: 2,
    total_tokens: 100,
    estimated_total_cost_usd: 0.00028,
    estimated_total_cost_twd: 0.00868,
  });

  assert.deepEqual(totals, {
    countedInputTokens: 200,
    promptTokens: 196,
    cachedInputTokens: 20,
    outputTokens: 50,
    thoughtsTokens: 6,
    totalTokens: 252,
    estimatedCostUsd: 0.0007,
    estimatedCostTwd: 0.0217,
  });
});

test("ignores missing and non-numeric usage fields", () => {
  const totals = emptyTranslationUsageTotals();
  addTranslationUsage(totals, null);
  addTranslationUsage(totals, {
    counted_input_tokens: "12",
    output_tokens: "not-a-number",
  });

  assert.equal(totals.countedInputTokens, 12);
  assert.equal(totals.outputTokens, 0);
});
