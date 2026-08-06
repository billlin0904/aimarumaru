"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  contentSignature,
  displayCueAt,
  translationForSourceId,
  validateDisplayCues,
} = require("../pages/subtitle_display_cues.js");

const group = {
  groupId: "g-41-44",
  sourceIds: [41, 42, 43, 44],
  segments: [
    { id: 41, start: 0, end: 2 },
    { id: 42, start: 2, end: 4 },
    { id: 43, start: 4, end: 6 },
    { id: 44, start: 6, end: 8 },
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
      translated_text: "而且，最令人驚訝的是，",
      lines: ["而且，最令人驚訝的是，"],
    },
    {
      cue_id: "g-41-44-c2",
      source_ids: [43, 44],
      start_ms: 4000,
      end_ms: 8000,
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
    translationForSourceId(cues, 44),
    "這架新望遠鏡已經提前完成，並且在預算內。",
  );
});

test("selects the cue by its own display timeline", () => {
  const cues = validateDisplayCues(group, result);
  assert.equal(displayCueAt(cues, 1.5).cueId, "g-41-44-c1");
  assert.equal(displayCueAt(cues, 4.0).cueId, "g-41-44-c2");
  assert.equal(displayCueAt(cues, 8.0), null);
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
