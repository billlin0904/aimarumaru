"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  contentSignature,
  displayCueAt,
  extractTitleTerms,
  normalizeSrtTimeline,
  progressiveSourceText,
  sourceTextForOverlay,
  sourceGroupContext,
  translationForSourceId,
  validateDisplayCues,
  withPrecedingSourceContext,
} = require("../pages/subtitle_display_cues.js");

test("attaches only the preceding source group as read-only context", () => {
  const previous = {
    groupId: "g-25-31",
    sourceIds: [25, 26, 27, 28, 29, 30, 31],
    sourceText: "OpenAI lost like 20-something billion dollars last year.",
    translatedText: "這個欄位不可傳給下一組",
  };
  const current = {
    groupId: "g-32-36",
    sourceIds: [32, 33, 34, 35, 36],
    sourceText: "Anthropic's probably not far behind.",
  };

  const context = sourceGroupContext(previous);
  const enriched = withPrecedingSourceContext(current, context);

  assert.deepEqual(enriched.precedingSourceContext, [{
    group_id: "g-25-31",
    source_ids: [25, 26, 27, 28, 29, 30, 31],
    source_text: "OpenAI lost like 20-something billion dollars last year.",
  }]);
  assert.equal(JSON.stringify(enriched).includes("這個欄位不可傳給下一組"), false);
  previous.sourceIds.push(99);
  assert.deepEqual(enriched.precedingSourceContext[0].source_ids, [25, 26, 27, 28, 29, 30, 31]);
});

test("extracts a work title without sending the whole YouTube title", () => {
  assert.deepEqual(
    extractTitleTerms("《聰明鎮》｜伊藤潤二觀看劇中角色｜Netflix"),
    ["聰明鎮"],
  );
  assert.deepEqual(
    extractTitleTerms("Bloody Smart | Netflix"),
    ["Bloody Smart"],
  );
});

const group = {
  groupId: "g-41-44",
  sourceIds: [41, 42, 43, 44],
  segments: [
    { id: 41, start: 0, end: 2, sourceText: "それがだいたい失敗して" },
    { id: 42, start: 2, end: 4, sourceText: "自分がひどい目に遭うという" },
    { id: 43, start: 4, end: 6, sourceText: "役柄の設定が" },
    { id: 44, start: 6, end: 8, sourceText: "面白いです。" },
  ],
};

const translatedText = "而且，最令人驚訝的是，這架新望遠鏡已經提前完成，並且在預算內。";
const result = {
  translated_text: translatedText,
  display_cues: [
    {
      cue_id: "g-41-44-c1",
      source_ids: [41, 42],
      start_ms: 0,
      end_ms: 4000,
      source_text: "それがだいたい失敗して自分がひどい目に遭うという",
      source_lines: ["それがだいたい失敗して自分がひどい目に遭うという"],
      translated_text: "而且，最令人驚訝的是，",
      lines: ["而且，最令人驚訝的是，"],
    },
    {
      cue_id: "g-41-44-c2",
      source_ids: [43, 44],
      start_ms: 4000,
      end_ms: 8000,
      source_text: "役柄の設定が面白いです。",
      source_lines: ["役柄の設定が面白いです。"],
      translated_text: "這架新望遠鏡已經提前完成，並且在預算內。",
      lines: ["這架新望遠鏡已經提前完成，", "並且在預算內。"],
    },
  ],
};

test("validates lossless display cues and exact source coverage", () => {
  const cues = validateDisplayCues(group, result);
  assert.equal(cues.length, 2);
  assert.equal(
    contentSignature(cues.map(cue => cue.translatedText).join("")),
    contentSignature(translatedText),
  );
  assert.equal(translationForSourceId(cues, 41), "而且，最令人驚訝的是，");
  assert.equal(
    cues[0].sourceText,
    "それがだいたい失敗して自分がひどい目に遭うという",
  );
  assert.equal(
    translationForSourceId(cues, 44),
    "這架新望遠鏡已經提前完成，並且在預算內。",
  );
});

test("selects the cue by its own display timeline", () => {
  const cues = validateDisplayCues(group, result);
  assert.equal(displayCueAt(cues, 1.5).cueId, "g-41-44-c1");
  assert.deepEqual(
    displayCueAt(cues, 1.5).sourceLines,
    ["それがだいたい失敗して自分がひどい目に遭うという"],
  );
  assert.equal(displayCueAt(cues, 4.0).cueId, "g-41-44-c2");
  assert.equal(displayCueAt(cues, 8.0), null);
});

test("keeps revealing source words after a translated display cue is ready", () => {
  const segment = {
    sourceText: "powerful as this one",
    words: [
      { word: "powerful", start: 1.0 },
      { word: " as", start: 1.5 },
      { word: " this", start: 2.0 },
      { word: " one", start: 2.5 },
    ],
  };
  const activeCue = {
    sourceLines: ["powerful as this one. But when these images come"],
  };

  assert.equal(progressiveSourceText(segment, 1.6), "powerful as");
  assert.equal(
    sourceTextForOverlay({
      activeCue,
      segment,
      currentTime: 1.6,
      revealWords: true,
    }),
    "powerful as",
  );
  assert.equal(
    sourceTextForOverlay({
      activeCue,
      segment,
      currentTime: 1.6,
      revealWords: false,
    }),
    "powerful as this one. But when these images come",
  );
});

test("derives source cue text while an older translation service is rolling over", () => {
  const legacyResult = structuredClone(result);
  for (const cue of legacyResult.display_cues) {
    delete cue.source_text;
    delete cue.source_lines;
  }

  const cues = validateDisplayCues(group, legacyResult);
  assert.equal(
    contentSignature(cues[0].sourceText),
    contentSignature(
      "それがだいたい失敗して自分がひどい目に遭うという",
    ),
  );
});

test("rejects duplicated full group translations", () => {
  const duplicated = structuredClone(result);
  duplicated.display_cues[0].translated_text = translatedText;
  duplicated.display_cues[1].translated_text = translatedText;
  assert.throws(
    () => validateDisplayCues(group, duplicated),
    /changed or duplicated translated content/,
  );
});

test("rejects gaps, overlaps, and missing source IDs", () => {
  const invalidTimeline = structuredClone(result);
  invalidTimeline.display_cues[1].start_ms = 4500;
  assert.throws(
    () => validateDisplayCues(group, invalidTimeline),
    /timeline must be contiguous/,
  );

  const missingSource = structuredClone(result);
  missingSource.display_cues[1].source_ids = [44];
  assert.throws(
    () => validateDisplayCues(group, missingSource),
    /cover source IDs exactly once/,
  );
});

test("removes only tiny SRT overlaps without changing cue starts", () => {
  const entries = normalizeSrtTimeline([
    { start: 4.4, end: 8.54, text: "first" },
    { start: 8.5, end: 10.0, text: "second" },
    { start: 9.5, end: 12.0, text: "intentional overlap" },
  ]);

  assert.equal(entries[0].end, 8.5);
  assert.equal(entries[1].start, 8.5);
  assert.equal(entries[1].end, 10.0);
  assert.equal(entries[2].start, 9.5);
});
