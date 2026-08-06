(function exposeSubtitleDisplayCues(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.SubtitleDisplayCues = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function createApi() {
  "use strict";

  function contentSignature(value) {
    return String(value ?? "").replace(/\s+/gu, "");
  }

  function validateDisplayCues(group, result) {
    const rawCues = result?.display_cues;
    if (!Array.isArray(rawCues) || rawCues.length === 0) return [];
    const segments = Array.isArray(group?.segments) ? group.segments : [];
    if (segments.length === 0) throw new Error("Display cues require source timeline segments");

    const expectedIds = Array.isArray(group.sourceIds) ? group.sourceIds.map(Number) : [];
    const validIds = new Set(expectedIds);
    const seenCueIds = new Set();
    const cues = rawCues.map((cue, index) => {
      const sourceIds = Array.isArray(cue?.source_ids) ? cue.source_ids.map(Number) : [];
      const assignedSegments = sourceIds.length > 0
        ? sourceIds.map(sourceId => segments.find(segment => Number(segment.id) === sourceId))
        : segments.filter(segment => (
          Number(segment.end) * 1000 > Number(cue?.start_ms)
          && Number(segment.start) * 1000 < Number(cue?.end_ms)
        ));
      const assignedSourceText = assignedSegments
        .map(segment => segment?.sourceText || "")
        .filter(Boolean)
        .join(" ")
        .trim();
      const sourceText = typeof cue?.source_text === "string" && cue.source_text.trim()
        ? cue.source_text.trim()
        : assignedSourceText;
      const returnedSourceLines = Array.isArray(cue?.source_lines)
        ? cue.source_lines.map(String)
        : [];
      const sourceLines = returnedSourceLines.length > 0
        ? returnedSourceLines
        : [sourceText];
      const lines = Array.isArray(cue?.lines) ? cue.lines.map(String) : [];
      if (
        typeof cue?.cue_id !== "string"
        || !cue.cue_id
        || seenCueIds.has(cue.cue_id)
        || !Number.isInteger(cue.start_ms)
        || !Number.isInteger(cue.end_ms)
        || cue.end_ms <= cue.start_ms
        || !sourceText
        || sourceLines.length < 1
        || sourceLines.length > 2
        || sourceLines.some(line => !line.trim())
        || typeof cue.translated_text !== "string"
        || !cue.translated_text.trim()
        || lines.length < 1
        || lines.length > 2
        || lines.some(line => !line.trim())
        || sourceIds.some(sourceId => !validIds.has(sourceId))
      ) {
        throw new Error(`Invalid display cue at index ${index}`);
      }
      seenCueIds.add(cue.cue_id);
      return {
        cueId: cue.cue_id,
        groupId: group.groupId,
        sourceIds,
        start: cue.start_ms / 1000,
        end: cue.end_ms / 1000,
        sourceText,
        sourceLines,
        translatedText: cue.translated_text.trim(),
        lines,
      };
    });

    const groupStart = Math.round(segments[0].start * 1000) / 1000;
    const groupEnd = Math.round(segments[segments.length - 1].end * 1000) / 1000;
    if (cues[0].start !== groupStart || cues[cues.length - 1].end !== groupEnd) {
      throw new Error("Display cues do not cover the complete group timeline");
    }
    for (let index = 1; index < cues.length; index += 1) {
      if (cues[index - 1].end !== cues[index].start) {
        throw new Error("Display cue timeline must be contiguous");
      }
    }

    const returnedIds = cues.flatMap(cue => cue.sourceIds);
    if (
      returnedIds.length !== expectedIds.length
      || returnedIds.some((sourceId, index) => sourceId !== expectedIds[index])
    ) {
      throw new Error("Display cues do not cover source IDs exactly once");
    }
    if (
      contentSignature(cues.map(cue => cue.translatedText).join(""))
      !== contentSignature(result.translated_text)
    ) {
      throw new Error("Display cues changed or duplicated translated content");
    }
    return cues;
  }

  function displayCueAt(cues, seconds) {
    let lower = 0;
    let upper = cues.length - 1;
    let candidate = null;
    while (lower <= upper) {
      const middle = Math.floor((lower + upper) / 2);
      const cue = cues[middle];
      if (cue.start <= seconds) {
        candidate = cue;
        lower = middle + 1;
      } else {
        upper = middle - 1;
      }
    }
    return candidate && seconds < candidate.end ? candidate : null;
  }

  function translationForSourceId(cues, sourceId) {
    const texts = cues
      .filter(cue => cue.sourceIds.includes(sourceId))
      .map(cue => cue.translatedText);
    return texts.join(" ").trim();
  }

  return {
    contentSignature,
    displayCueAt,
    translationForSourceId,
    validateDisplayCues,
  };
}));
