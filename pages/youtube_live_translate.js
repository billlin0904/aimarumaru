"use strict";

const form = document.getElementById("translateForm");
const youtubeUrl = document.getElementById("youtubeUrl");
const ignoreSubtitles = document.getElementById("ignoreSubtitles");
const sourceLanguage = document.getElementById("sourceLanguage");
const targetLanguage = document.getElementById("targetLanguage");
const startBtn = document.getElementById("startBtn");
const languageConfirmation = document.getElementById("languageConfirmation");
const detectedLanguageTitle = document.getElementById("detectedLanguageTitle");
const languageConfidence = document.getElementById("languageConfidence");
const confirmedSourceLanguage = document.getElementById("confirmedSourceLanguage");
const confirmLanguageBtn = document.getElementById("confirmLanguageBtn");
const videoPreview = document.getElementById("videoPreview");
const videoPlayerLayer = document.getElementById("videoPlayerLayer");
const videoSubtitleOverlay = document.getElementById("videoSubtitleOverlay");
const videoSubtitleSource = document.getElementById("videoSubtitleSource");
const videoSubtitleTranslation = document.getElementById("videoSubtitleTranslation");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const videoMeta = document.getElementById("videoMeta");
const videoTitle = document.getElementById("videoTitle");
const videoDetail = document.getElementById("videoDetail");
const resultPanel = document.getElementById("resultPanel");
const progressGrid = document.getElementById("progressGrid");
const transcriptionPercent = document.getElementById("transcriptionPercent");
const transcriptionBar = document.getElementById("transcriptionBar");
const transcriptionDetail = document.getElementById("transcriptionDetail");
const translationPercent = document.getElementById("translationPercent");
const translationBar = document.getElementById("translationBar");
const translationDetail = document.getElementById("translationDetail");
const segmentList = document.getElementById("segmentList");
const segmentCount = document.getElementById("segmentCount");
const emptyState = document.getElementById("emptyState");
const actionBar = document.getElementById("actionBar");
const downloadSourceSrt = document.getElementById("downloadSourceSrt");
const downloadTranslatedSrt = document.getElementById("downloadTranslatedSrt");
const downloadBilingualSrt = document.getElementById("downloadBilingualSrt");
const downloadSegmentsJson = document.getElementById("downloadSegmentsJson");
const nextTranscription = document.getElementById("nextTranscription");
const captchaBlock = document.getElementById("captchaBlock");
const captchaImage = document.getElementById("captchaImage");
const captchaId = document.getElementById("captchaId");
const captchaToken = document.getElementById("captchaToken");
const captchaAnswer = document.getElementById("captchaAnswer");
const refreshCaptcha = document.getElementById("refreshCaptcha");
const verifyCaptcha = document.getElementById("verifyCaptcha");
const captchaStatus = document.getElementById("captchaStatus");

const BATCH_DELAY_MS = 2000;
const BATCH_SEGMENT_TRIGGER = 8;
const BATCH_CHARACTER_TRIGGER = 2000;
const BATCH_MAX_SEGMENTS = 40;
const BATCH_MAX_CHARACTERS = 2000;
const REQUEST_MAX_CHARACTERS = 6000;
const CONTEXT_MAX_SEGMENTS = 5;
const RETRY_DELAYS_MS = [2000, 5000];
const RETRYABLE_HTTP_STATUSES = new Set([429, 502, 503, 504]);
const PROMPT_VERSION = "subtitle-v1";

const params = new URLSearchParams(window.location.search);
const languageStorageKey = "audioTranscribeLanguage";
const translations = {
  "zh-Hant": {
    pageTitle: "Video Translate", transcribeOnly: "僅轉譯", urlLabel: "YouTube 網址", ignoreSubtitles: "忽略內建字幕",
    videoPreview: "影片預覽", subtitleWaiting: "字幕完成後會顯示在這裡",
    sourceLanguage: "原文語言", targetLanguage: "翻譯目標語言", autoDetect: "自動偵測", languageEnglish: "英文", languageJapanese: "日文", languageKorean: "韓文", languageThai: "泰文", languageTraditionalChinese: "繁體中文",
    start: "開始轉譯並翻譯", detectLanguage: "偵測語言", waiting: "等待輸入網址", creating: "建立任務中", processing: "處理中", done: "轉譯與翻譯完成", partialDone: "處理完成，部分翻譯失敗", failed: "處理失敗", disconnected: "連線中斷", requestFailed: "請求失敗",
    detectedLanguageLabel: "偵測結果", detectedLanguageValue: "偵測為 {language}", confirmSourceLanguage: "確認原文語言", confirmAndStart: "確認並開始", confirmDetectedLanguage: "請確認原文語言後再開始轉譯", confirmingLanguage: "正在送出語言選擇", confidence: "偵測信心 {percent}%", subtitleLanguageSource: "來自 YouTube 字幕語言", unknownLanguage: "未知語言",
    resultTitle: "即時字幕", emptyState: "原文與譯文會一段一段顯示在這裡。", segmentUnit: "段", sourceText: "原文", translatedText: "翻譯", translationPending: "正在等待翻譯…", translationFailed: "翻譯失敗", retryTranslation: "重新翻譯",
    transcriptionProgress: "轉譯進度", translationProgress: "翻譯進度", progressDetail: "取得影片長度後會顯示轉譯進度。", translationWaiting: "等待轉譯內容。", estimatingCompletion: "正在估算完成時間…", estimatedCompletion: "預計完成時間 {time}", durationPrefix: "長度", seconds: "秒",
    translationCounts: "已翻譯 {translated} / 已收到 {received} 段 · 等待翻譯 {waiting} 段", translationPercentDone: "翻譯完成 {percent}% · 失敗 {failed} 段", translationErrorCodes: "錯誤碼 {codes}", sameLanguage: "原文與目標語言相同，已略過翻譯。", unsupportedLanguage: "目前翻譯服務不支援偵測到的語言：{language}",
    downloadSourceSrt: "下載原始 SRT", downloadTranslatedSrt: "下載翻譯 SRT", downloadBilingualSrt: "下載雙語 SRT", downloadJson: "下載 segments JSON", partialSuffix: "（部分完成）", partialDownloadWarning: "部分段落尚未翻譯，下載檔將以原文補位。是否繼續？", nextTranscription: "下一個轉譯內容",
    captchaLabel: "驗證碼", captchaPlaceholder: "輸入圖片中的文字", refreshCaptcha: "重新選擇", verifyCaptcha: "驗證", captchaVerified: "驗證完成", captchaLoadFailed: "取得驗證碼失敗", captchaVerifyFailed: "驗證失敗", captchaRequired: "請先完成驗證碼",
    translationServiceFailed: "翻譯服務暫時無法使用", invalidTranslation: "翻譯回應格式不正確", videoTitle: "YouTube 影片", about: "關於我們", privacy: "隱私權政策", terms: "使用條款", contact: "聯絡我們", leaveWarning: "轉譯或翻譯仍在進行中，確定要離開頁面嗎？"
  },
  en: {
    pageTitle: "Video Translate", transcribeOnly: "Transcribe only", urlLabel: "YouTube URL", ignoreSubtitles: "Ignore built-in subtitles",
    videoPreview: "Video preview", subtitleWaiting: "Subtitles will appear here when ready",
    sourceLanguage: "Source language", targetLanguage: "Target language", autoDetect: "Auto detect", languageEnglish: "English", languageJapanese: "Japanese", languageKorean: "Korean", languageThai: "Thai", languageTraditionalChinese: "Traditional Chinese",
    start: "Transcribe and translate", detectLanguage: "Detect language", waiting: "Waiting for a URL", creating: "Creating job", processing: "Processing", done: "Transcription and translation complete", partialDone: "Complete with some translation failures", failed: "Processing failed", disconnected: "Connection interrupted", requestFailed: "Request failed",
    detectedLanguageLabel: "Detection result", detectedLanguageValue: "Detected as {language}", confirmSourceLanguage: "Confirm source language", confirmAndStart: "Confirm and start", confirmDetectedLanguage: "Confirm the source language to begin transcription", confirmingLanguage: "Submitting language selection", confidence: "Detection confidence {percent}%", subtitleLanguageSource: "From YouTube subtitle language", unknownLanguage: "Unknown language",
    resultTitle: "Live subtitles", emptyState: "Source text and translation will appear here segment by segment.", segmentUnit: "segments", sourceText: "Source", translatedText: "Translation", translationPending: "Waiting for translation…", translationFailed: "Translation failed", retryTranslation: "Retry",
    transcriptionProgress: "Transcription", translationProgress: "Translation", progressDetail: "Progress appears after the video duration is available.", translationWaiting: "Waiting for transcription.", estimatingCompletion: "Estimating completion time…", estimatedCompletion: "Estimated completion {time}", durationPrefix: "Duration", seconds: "sec",
    translationCounts: "Translated {translated} / {received} received · {waiting} waiting", translationPercentDone: "Translation {percent}% · {failed} failed", translationErrorCodes: "Error code {codes}", sameLanguage: "Source and target languages match. Translation was skipped.", unsupportedLanguage: "The translation service does not support the detected language: {language}",
    downloadSourceSrt: "Download source SRT", downloadTranslatedSrt: "Download translated SRT", downloadBilingualSrt: "Download bilingual SRT", downloadJson: "Download segments JSON", partialSuffix: " (partial)", partialDownloadWarning: "Some segments are not translated. Source text will be used as fallback. Continue?", nextTranscription: "Next transcription",
    captchaLabel: "Verification", captchaPlaceholder: "Enter the text in the image", refreshCaptcha: "Choose again", verifyCaptcha: "Verify", captchaVerified: "Verified", captchaLoadFailed: "Could not load verification image", captchaVerifyFailed: "Verification failed", captchaRequired: "Please complete verification first",
    translationServiceFailed: "Translation service is temporarily unavailable", invalidTranslation: "The translation response is invalid", videoTitle: "YouTube video", about: "About", privacy: "Privacy", terms: "Terms", contact: "Contact", leaveWarning: "Transcription or translation is still running. Leave this page?"
  },
  ja: {
    pageTitle: "Video Translate", transcribeOnly: "文字起こしのみ", urlLabel: "YouTube URL", ignoreSubtitles: "内蔵字幕を無視",
    videoPreview: "動画プレビュー", subtitleWaiting: "字幕の準備ができるとここに表示されます",
    sourceLanguage: "原文の言語", targetLanguage: "翻訳先の言語", autoDetect: "自動検出", languageEnglish: "英語", languageJapanese: "日本語", languageKorean: "韓国語", languageThai: "タイ語", languageTraditionalChinese: "繁体字中国語",
    start: "文字起こしと翻訳を開始", detectLanguage: "言語を検出", waiting: "URL を入力してください", creating: "ジョブを作成中", processing: "処理中", done: "文字起こしと翻訳が完了しました", partialDone: "一部の翻訳に失敗しました", failed: "処理に失敗しました", disconnected: "接続が切断されました", requestFailed: "リクエストに失敗しました",
    detectedLanguageLabel: "検出結果", detectedLanguageValue: "{language} として検出", confirmSourceLanguage: "原文の言語を確認", confirmAndStart: "確認して開始", confirmDetectedLanguage: "原文の言語を確認してから開始してください", confirmingLanguage: "言語設定を送信中", confidence: "検出の信頼度 {percent}%", subtitleLanguageSource: "YouTube 字幕の言語", unknownLanguage: "不明な言語",
    resultTitle: "リアルタイム字幕", emptyState: "原文と翻訳が順番に表示されます。", segmentUnit: "件", sourceText: "原文", translatedText: "翻訳", translationPending: "翻訳待ち…", translationFailed: "翻訳に失敗しました", retryTranslation: "再翻訳",
    transcriptionProgress: "文字起こしの進捗", translationProgress: "翻訳の進捗", progressDetail: "動画の長さを取得後、進捗が表示されます。", translationWaiting: "文字起こしを待っています。", estimatingCompletion: "完了時刻を計算中…", estimatedCompletion: "完了予定 {time}", durationPrefix: "長さ", seconds: "秒",
    translationCounts: "翻訳済み {translated} / 受信 {received} 件・待機 {waiting} 件", translationPercentDone: "翻訳 {percent}%・失敗 {failed} 件", translationErrorCodes: "エラーコード {codes}", sameLanguage: "原文と翻訳先が同じため、翻訳を省略しました。", unsupportedLanguage: "検出された言語は現在サポートされていません：{language}",
    downloadSourceSrt: "原文 SRT をダウンロード", downloadTranslatedSrt: "翻訳 SRT をダウンロード", downloadBilingualSrt: "二言語 SRT をダウンロード", downloadJson: "segments JSON をダウンロード", partialSuffix: "（一部完了）", partialDownloadWarning: "未翻訳の区間は原文で補完されます。続行しますか？", nextTranscription: "次の文字起こし",
    captchaLabel: "認証コード", captchaPlaceholder: "画像内の文字を入力", refreshCaptcha: "選び直す", verifyCaptcha: "認証", captchaVerified: "認証完了", captchaLoadFailed: "認証画像を取得できませんでした", captchaVerifyFailed: "認証に失敗しました", captchaRequired: "先に認証を完了してください",
    translationServiceFailed: "翻訳サービスを利用できません", invalidTranslation: "翻訳レスポンスが不正です", videoTitle: "YouTube 動画", about: "私たちについて", privacy: "プライバシー", terms: "利用規約", contact: "お問い合わせ", leaveWarning: "文字起こしまたは翻訳が進行中です。ページを離れますか？"
  },
  ko: {
    pageTitle: "Video Translate", transcribeOnly: "전사만", urlLabel: "YouTube URL", ignoreSubtitles: "내장 자막 무시",
    videoPreview: "동영상 미리보기", subtitleWaiting: "자막이 준비되면 여기에 표시됩니다",
    sourceLanguage: "원문 언어", targetLanguage: "번역 언어", autoDetect: "자동 감지", languageEnglish: "영어", languageJapanese: "일본어", languageKorean: "한국어", languageThai: "태국어", languageTraditionalChinese: "번체 중국어",
    start: "전사 및 번역 시작", detectLanguage: "언어 감지", waiting: "URL 입력 대기 중", creating: "작업 생성 중", processing: "처리 중", done: "전사 및 번역 완료", partialDone: "일부 번역 실패와 함께 완료", failed: "처리 실패", disconnected: "연결이 끊겼습니다", requestFailed: "요청 실패",
    detectedLanguageLabel: "감지 결과", detectedLanguageValue: "{language}(으)로 감지", confirmSourceLanguage: "원문 언어 확인", confirmAndStart: "확인 후 시작", confirmDetectedLanguage: "원문 언어를 확인한 뒤 전사를 시작하세요", confirmingLanguage: "언어 선택을 전송하는 중", confidence: "감지 신뢰도 {percent}%", subtitleLanguageSource: "YouTube 자막 언어", unknownLanguage: "알 수 없는 언어",
    resultTitle: "실시간 자막", emptyState: "원문과 번역이 구간별로 표시됩니다.", segmentUnit: "개", sourceText: "원문", translatedText: "번역", translationPending: "번역 대기 중…", translationFailed: "번역 실패", retryTranslation: "다시 번역",
    transcriptionProgress: "전사 진행률", translationProgress: "번역 진행률", progressDetail: "영상 길이를 가져오면 진행률이 표시됩니다.", translationWaiting: "전사 내용을 기다리는 중입니다.", estimatingCompletion: "완료 시간을 계산하는 중…", estimatedCompletion: "예상 완료 시간 {time}", durationPrefix: "길이", seconds: "초",
    translationCounts: "번역 {translated} / 수신 {received}개 · 대기 {waiting}개", translationPercentDone: "번역 {percent}% · 실패 {failed}개", translationErrorCodes: "오류 코드 {codes}", sameLanguage: "원문과 대상 언어가 같아 번역을 건너뛰었습니다.", unsupportedLanguage: "감지된 언어는 현재 지원되지 않습니다: {language}",
    downloadSourceSrt: "원문 SRT 다운로드", downloadTranslatedSrt: "번역 SRT 다운로드", downloadBilingualSrt: "이중 언어 SRT 다운로드", downloadJson: "segments JSON 다운로드", partialSuffix: " (일부 완료)", partialDownloadWarning: "번역되지 않은 구간은 원문으로 대체됩니다. 계속하시겠습니까?", nextTranscription: "다음 전사",
    captchaLabel: "인증 코드", captchaPlaceholder: "이미지의 문자를 입력하세요", refreshCaptcha: "다시 선택", verifyCaptcha: "인증", captchaVerified: "인증 완료", captchaLoadFailed: "인증 이미지를 불러오지 못했습니다", captchaVerifyFailed: "인증에 실패했습니다", captchaRequired: "먼저 인증을 완료해 주세요",
    translationServiceFailed: "번역 서비스를 일시적으로 사용할 수 없습니다", invalidTranslation: "번역 응답이 올바르지 않습니다", videoTitle: "YouTube 영상", about: "소개", privacy: "개인정보 처리방침", terms: "이용약관", contact: "문의", leaveWarning: "전사 또는 번역이 진행 중입니다. 페이지를 떠나시겠습니까?"
  }
};

let currentLanguage = translations[params.get("lang")]
  ? params.get("lang")
  : localStorage.getItem(languageStorageKey) || "zh-Hant";
if (!translations[currentLanguage]) currentLanguage = "zh-Hant";
if (params.get("embedded") === "1") document.body.classList.add("embedded");

const sourceSegments = new Map();
const translatedSegments = new Map();
const segmentNodes = new Map();
const failedSegmentIds = new Set();
const translationFailureCodes = new Map();
let pendingSegments = [];
let eventSource = null;
let translationQueue = Promise.resolve();
let batchTimer = null;
let queuedBatchCount = 0;
let batchCounter = 0;
let currentJobId = "";
let currentTranslationToken = "";
let requestedSourceLanguage = "";
let selectedTargetLanguage = "zh-TW";
let detectedSourceLanguage = "";
let totalDuration = 0;
let transcriptionDone = false;
let transcriptionFailed = false;
let transcriptionActive = false;
let translationActive = false;
let captchaEnabled = true;
let sameLanguageNoticeShown = false;
let unsupportedLanguageNotice = "";
let previewTimer = null;
let currentPreviewKey = "";
let youtubePlayerApiPromise = null;
let youtubePlayerController = null;
let playbackSyncTimer = null;
let playbackSyncEnabled = false;
let activePlaybackSegmentId = null;
let detectedLanguageData = null;

function t(key, replacements = {}) {
  let value = translations[currentLanguage]?.[key]
    || translations["zh-Hant"][key]
    || key;
  for (const [name, replacement] of Object.entries(replacements)) {
    value = value.replaceAll(`{${name}}`, String(replacement));
  }
  return value;
}

function applyLanguage(languageKey) {
  currentLanguage = translations[languageKey] ? languageKey : "zh-Hant";
  localStorage.setItem(languageStorageKey, currentLanguage);
  document.documentElement.lang = currentLanguage;
  document.title = t("pageTitle");
  document.querySelectorAll("[data-i18n]").forEach(element => {
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach(element => {
    element.setAttribute("placeholder", t(element.dataset.i18nPlaceholder));
  });
  document.querySelectorAll("[data-i18n-alt]").forEach(element => {
    element.setAttribute("alt", t(element.dataset.i18nAlt));
  });
  const playerFrame = document.getElementById("youtubePlayer");
  if (playerFrame) playerFrame.setAttribute("title", t("videoPreview"));
  updateStartButton();
  if (detectedLanguageData) showLanguageConfirmation(detectedLanguageData);
  updateTranslationProgress();
  updateActionButtons();
}

function parseYouTubeStartTime(value) {
  const text = String(value || "").trim().toLowerCase();
  if (!text) return 0;
  if (/^\d+$/.test(text)) return Number(text);
  const match = text.match(/^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$/);
  if (!match) return 0;
  return (Number(match[1]) || 0) * 3600
    + (Number(match[2]) || 0) * 60
    + (Number(match[3]) || 0);
}

function parseYouTubeVideo(value) {
  let url;
  try {
    url = new URL(String(value || "").trim());
  } catch (_) {
    return null;
  }

  const host = url.hostname.toLowerCase().replace(/^www\./, "");
  const pathParts = url.pathname.split("/").filter(Boolean);
  let videoId = "";
  if (host === "youtu.be") {
    videoId = pathParts[0] || "";
  } else if (host === "youtube.com" || host.endsWith(".youtube.com")) {
    if (url.pathname === "/watch") {
      videoId = url.searchParams.get("v") || "";
    } else if (["shorts", "live", "embed", "v"].includes(pathParts[0])) {
      videoId = pathParts[1] || "";
    }
  }
  if (!/^[A-Za-z0-9_-]{11}$/.test(videoId)) return null;

  const hashParams = new URLSearchParams(url.hash.replace(/^#/, ""));
  const start = parseYouTubeStartTime(
    url.searchParams.get("t")
      || url.searchParams.get("start")
      || hashParams.get("t"),
  );
  return { videoId, start };
}

function loadYouTubePlayerApi() {
  if (window.YT?.Player) return Promise.resolve(window.YT);
  if (youtubePlayerApiPromise) return youtubePlayerApiPromise;

  youtubePlayerApiPromise = new Promise((resolve, reject) => {
    const previousReadyHandler = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      if (typeof previousReadyHandler === "function") previousReadyHandler();
      resolve(window.YT);
    };

    if (!document.querySelector("script[data-youtube-iframe-api]")) {
      const script = document.createElement("script");
      script.src = "https://www.youtube.com/iframe_api";
      script.dataset.youtubeIframeApi = "true";
      script.onerror = () => reject(new Error("Could not load YouTube Player API"));
      document.head.append(script);
    }
  });
  return youtubePlayerApiPromise;
}

function stopPlaybackSync() {
  if (playbackSyncTimer) clearInterval(playbackSyncTimer);
  playbackSyncTimer = null;
  playbackSyncEnabled = false;
  activePlaybackSegmentId = null;
  document.querySelector(".segment-card.active")?.classList.remove("active");
  clearVideoSubtitleOverlay();
}

function clearVideoSubtitleOverlay(showWaiting = true) {
  videoSubtitleSource.textContent = "";
  videoSubtitleTranslation.textContent = showWaiting ? t("subtitleWaiting") : "";
  videoSubtitleSource.classList.add("d-none");
  videoSubtitleOverlay.classList.toggle("waiting", showWaiting);
  videoSubtitleOverlay.classList.remove("error");
}

function updateVideoSubtitleOverlay(segment) {
  const failureCode = segment
    ? translationFailureCodes.get(segment.id)
    : "";
  if (segment && failureCode) {
    videoSubtitleSource.textContent = "";
    videoSubtitleSource.classList.add("d-none");
    videoSubtitleTranslation.textContent = `${t("translationFailed")} · ${t(
      "translationErrorCodes",
      { codes: failureCode },
    )}`;
    videoSubtitleOverlay.classList.remove("waiting");
    videoSubtitleOverlay.classList.add("error");
    return;
  }

  const translatedText = segment
    ? translatedSegments.get(segment.id)?.translatedText
    : "";
  if (!segment) {
    clearVideoSubtitleOverlay(false);
    return;
  }
  if (!translatedText) {
    clearVideoSubtitleOverlay();
    return;
  }

  videoSubtitleSource.textContent = segment.sourceText;
  videoSubtitleTranslation.textContent = translatedText;
  videoSubtitleSource.classList.toggle(
    "d-none",
    segment.sourceText.trim() === translatedText.trim(),
  );
  videoSubtitleOverlay.classList.remove("waiting");
  videoSubtitleOverlay.classList.remove("error");
}

function resetVideoPlayerMount() {
  stopPlaybackSync();
  if (youtubePlayerController) {
    try {
      youtubePlayerController.destroy();
    } catch (_) {}
  }
  youtubePlayerController = null;
  videoPlayerLayer.replaceChildren();
  const mount = document.createElement("div");
  mount.id = "youtubePlayer";
  mount.setAttribute("title", t("videoPreview"));
  videoPlayerLayer.append(mount);
  return mount;
}

function playbackSegmentAt(seconds) {
  for (const segment of sortedSourceSegments()) {
    if (segment.start > seconds) break;
    if (
      (translatedSegments.has(segment.id) || failedSegmentIds.has(segment.id))
      && seconds >= segment.start
      && seconds < segment.end
    ) {
      return segment;
    }
  }
  return null;
}

function focusPlaybackSegment(segment) {
  updateVideoSubtitleOverlay(segment);
  const nextId = segment?.id || null;
  if (nextId === activePlaybackSegmentId) return;
  if (activePlaybackSegmentId) {
    segmentNodes.get(activePlaybackSegmentId)?.card.classList.remove("active");
  }
  activePlaybackSegmentId = nextId;
  if (!segment) return;

  const nodes = segmentNodes.get(segment.id);
  if (!nodes || nodes.card.classList.contains("d-none")) return;
  nodes.card.classList.add("active");
  const listRect = segmentList.getBoundingClientRect();
  const cardRect = nodes.card.getBoundingClientRect();
  const targetTop = segmentList.scrollTop
    + cardRect.top
    - listRect.top
    - (listRect.height - cardRect.height) / 2;
  segmentList.scrollTop = Math.max(0, targetTop);
}

function syncPlaybackToSubtitles() {
  if (!playbackSyncEnabled || !youtubePlayerController?.getCurrentTime) return;
  try {
    focusPlaybackSegment(playbackSegmentAt(youtubePlayerController.getCurrentTime()));
  } catch (_) {}
}

function startPlaybackSync() {
  if (playbackSyncTimer) clearInterval(playbackSyncTimer);
  playbackSyncTimer = setInterval(syncPlaybackToSubtitles, 400);
}

async function renderYouTubePlayer(parsed, previewKey) {
  const mount = resetVideoPlayerMount();
  let YT;
  try {
    YT = await loadYouTubePlayerApi();
  } catch (_) {
    return;
  }
  if (previewKey !== currentPreviewKey || !mount.isConnected) return;

  youtubePlayerController = new YT.Player(mount, {
    host: "https://www.youtube-nocookie.com",
    videoId: parsed.videoId,
    playerVars: {
      start: parsed.start,
      rel: 0,
      playsinline: 1,
      origin: window.location.origin,
    },
    events: {
      onReady: event => {
        if (previewKey !== currentPreviewKey) return;
        event.target.getIframe().setAttribute("title", t("videoPreview"));
        startPlaybackSync();
        notifyParentHeight();
      },
      onStateChange: event => {
        if (event.data === YT.PlayerState.UNSTARTED || event.data === YT.PlayerState.CUED) return;
        playbackSyncEnabled = true;
        syncPlaybackToSubtitles();
      },
    },
  });
}

function updateVideoPreview() {
  const parsed = parseYouTubeVideo(youtubeUrl.value);
  if (!parsed) {
    currentPreviewKey = "";
    resetVideoPlayerMount();
    videoPreview.classList.add("d-none");
    notifyParentHeight();
    return false;
  }

  const previewKey = `${parsed.videoId}:${parsed.start}`;
  if (previewKey !== currentPreviewKey) {
    currentPreviewKey = previewKey;
    renderYouTubePlayer(parsed, previewKey);
  }
  videoPreview.classList.remove("d-none");
  notifyParentHeight();
  return true;
}

function scheduleVideoPreview() {
  if (previewTimer) clearTimeout(previewTimer);
  previewTimer = setTimeout(updateVideoPreview, 300);
}

function normalizeTranslationLanguage(language) {
  const value = String(language || "").toLowerCase();
  if (value.startsWith("en")) return "en";
  if (value.startsWith("ja")) return "ja";
  if (value.startsWith("ko")) return "ko";
  if (value.startsWith("th")) return "th";
  if (value.startsWith("zh")) return "zh-TW";
  return null;
}

function languageName(language) {
  const names = {
    en: "languageEnglish",
    ja: "languageJapanese",
    ko: "languageKorean",
    th: "languageThai",
    "zh-TW": "languageTraditionalChinese",
  };
  return names[language] ? t(names[language]) : t("unknownLanguage");
}

function updateStartButton() {
  startBtn.textContent = sourceLanguage.value ? t("start") : t("detectLanguage");
}

function revealResults() {
  resultPanel.classList.remove("d-none");
  notifyParentHeight();
}

function showLanguageConfirmation(data) {
  detectedLanguageData = data;
  const normalized = normalizeTranslationLanguage(data.language);
  const displayLanguage = normalized
    ? languageName(normalized)
    : String(data.language || t("unknownLanguage"));
  detectedLanguageTitle.textContent = t("detectedLanguageValue", { language: displayLanguage });
  confirmedSourceLanguage.value = normalized || "";
  confirmLanguageBtn.disabled = !normalized;

  const hasProbability = data.language_probability !== null
    && data.language_probability !== undefined
    && data.language_probability !== "";
  const probability = hasProbability ? Number(data.language_probability) : Number.NaN;
  languageConfidence.classList.remove("medium", "low");
  if (data.source === "youtube_subtitles") {
    languageConfidence.textContent = t("subtitleLanguageSource");
  } else if (Number.isFinite(probability)) {
    const percent = Math.round(Math.max(0, Math.min(1, probability)) * 100);
    languageConfidence.textContent = t("confidence", { percent });
    if (percent < 60) languageConfidence.classList.add("low");
    else if (percent < 80) languageConfidence.classList.add("medium");
  } else {
    languageConfidence.textContent = t("unknownLanguage");
    languageConfidence.classList.add("low");
  }
  languageConfirmation.classList.remove("d-none");
  setStatus(t("confirmDetectedLanguage"), "idle");
  notifyParentHeight();
}

function setStatus(message, state = "idle") {
  statusText.textContent = message;
  statusDot.classList.toggle("running", state === "running");
  statusDot.classList.toggle("failed", state === "failed");
  notifyParentHeight();
}

function notifyParentHeight() {
  if (params.get("embedded") !== "1" || !window.parent) return;
  requestAnimationFrame(() => {
    const height = Math.ceil(document.documentElement.scrollHeight);
    window.parent.postMessage({ type: "youtube-live-height", height }, window.location.origin);
  });
}

function updateBusyState() {
  const active = transcriptionActive || translationActive;
  if (params.get("embedded") === "1" && window.parent) {
    window.parent.postMessage({ type: "youtube-live-busy", active }, window.location.origin);
  }
}

function setTranscriptionActive(active) {
  transcriptionActive = active;
  updateBusyState();
}

function setTranslationActive(active) {
  translationActive = active;
  updateBusyState();
}

window.addEventListener("beforeunload", event => {
  if (!transcriptionActive && !translationActive) return;
  event.preventDefault();
  event.returnValue = t("leaveWarning");
  return event.returnValue;
});

function formatDisplayTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const secs = Math.floor(value % 60);
  return hours > 0
    ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function formatSrtTimestamp(seconds) {
  const millisTotal = Math.max(0, Math.round((Number(seconds) || 0) * 1000));
  const hours = Math.floor(millisTotal / 3600000);
  const minutes = Math.floor((millisTotal % 3600000) / 60000);
  const secs = Math.floor((millisTotal % 60000) / 1000);
  const millis = millisTotal % 1000;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")},${String(millis).padStart(3, "0")}`;
}

function sortedSourceSegments() {
  return [...sourceSegments.values()].sort((left, right) => left.id - right.id);
}

function resetView() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  if (batchTimer) {
    clearTimeout(batchTimer);
    batchTimer = null;
  }
  sourceSegments.clear();
  translatedSegments.clear();
  segmentNodes.clear();
  failedSegmentIds.clear();
  translationFailureCodes.clear();
  pendingSegments = [];
  translationQueue = Promise.resolve();
  queuedBatchCount = 0;
  batchCounter = 0;
  currentJobId = "";
  currentTranslationToken = "";
  detectedSourceLanguage = "";
  detectedLanguageData = null;
  totalDuration = 0;
  transcriptionDone = false;
  transcriptionFailed = false;
  sameLanguageNoticeShown = false;
  unsupportedLanguageNotice = "";
  activePlaybackSegmentId = null;
  clearVideoSubtitleOverlay();
  setTranscriptionActive(false);
  setTranslationActive(false);
  segmentList.replaceChildren(emptyState);
  segmentList.scrollTop = 0;
  emptyState.classList.remove("d-none");
  segmentCount.textContent = `0 ${t("segmentUnit")}`;
  progressGrid.classList.add("d-none");
  actionBar.classList.add("d-none");
  languageConfirmation.classList.add("d-none");
  resultPanel.classList.add("d-none");
  videoMeta.classList.add("d-none");
  videoTitle.textContent = "";
  videoDetail.textContent = "";
  setTranscriptionProgress(0);
  updateTranslationProgress();
  notifyParentHeight();
}

function setTranscriptionProgress(percent, detail = null) {
  const value = Math.max(0, Math.min(100, Math.round(Number(percent) || 0)));
  transcriptionPercent.textContent = `${value}%`;
  transcriptionBar.style.width = `${value}%`;
  transcriptionBar.parentElement.setAttribute("aria-valuenow", String(value));
  if (detail !== null) transcriptionDetail.textContent = detail;
}

function updateTranscriptionProgress(data) {
  const serverPercent = Number(data.progress_percent);
  const fallbackPercent = totalDuration > 0
    ? (Number(data.end) || 0) / totalDuration * 100
    : 0;
  setTranscriptionProgress(
    Number.isFinite(serverPercent) ? serverPercent : fallbackPercent,
  );
  if (Object.prototype.hasOwnProperty.call(data, "estimated_completion_at")) {
    if (data.estimated_completion_at) {
      const completionTime = new Intl.DateTimeFormat(currentLanguage, {
        hour: "2-digit",
        minute: "2-digit",
      }).format(new Date(data.estimated_completion_at));
      transcriptionDetail.textContent = t("estimatedCompletion", { time: completionTime });
    } else {
      transcriptionDetail.textContent = t("estimatingCompletion");
    }
  }
}

function unresolvedCount() {
  return Math.max(0, sourceSegments.size - translatedSegments.size - failedSegmentIds.size);
}

function translationErrorCode(error) {
  const codes = [];
  if (Number.isInteger(error?.status)) codes.push(`HTTP ${error.status}`);
  if (error?.code) {
    const code = String(error.code).trim();
    if (code && !codes.includes(code)) codes.push(code);
  }
  return codes.join(" / ") || "TRANSLATION_ERROR";
}

function currentTranslationErrorCodes() {
  return [...new Set(translationFailureCodes.values())].sort();
}

function appendTranslationErrorCodes(message) {
  const codes = currentTranslationErrorCodes();
  return codes.length > 0
    ? `${message} · ${t("translationErrorCodes", { codes: codes.join(", ") })}`
    : message;
}

function updateTranslationProgress() {
  const received = sourceSegments.size;
  const translated = translatedSegments.size;
  const failed = failedSegmentIds.size;
  const waiting = unresolvedCount();
  let percent = received > 0 ? translated / received * 100 : 0;

  if (transcriptionDone) {
    const rounded = Math.max(0, Math.min(100, Math.round(percent)));
    translationPercent.textContent = `${rounded}%`;
    translationDetail.textContent = appendTranslationErrorCodes(t("translationPercentDone", {
      percent: rounded,
      failed,
    }));
  } else {
    translationPercent.textContent = received > 0 ? `${translated}/${received}` : "—";
    const detail = received > 0
      ? t("translationCounts", { translated, received, waiting })
      : t("translationWaiting");
    translationDetail.textContent = appendTranslationErrorCodes(detail);
  }

  translationBar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  translationBar.parentElement.setAttribute("aria-valuenow", String(Math.round(percent)));
  segmentCount.textContent = `${translated} ${t("segmentUnit")}`;
}

function scrollSegmentsToLatest() {
  requestAnimationFrame(() => {
    segmentList.scrollTop = segmentList.scrollHeight;
  });
}

function renderSegment(segment) {
  const card = document.createElement("article");
  card.className = "segment-card d-none";
  card.dataset.segmentId = String(segment.id);

  const meta = document.createElement("div");
  meta.className = "segment-meta";
  const time = document.createElement("button");
  time.className = "segment-time";
  time.type = "button";
  time.dataset.seekTime = String(segment.start);
  time.textContent = `${formatDisplayTime(segment.start)} → ${formatDisplayTime(segment.end)}`;
  const number = document.createElement("span");
  number.textContent = `#${segment.id}`;
  meta.append(time, number);

  const copy = document.createElement("div");
  copy.className = "segment-copy";
  const sourceColumn = document.createElement("div");
  sourceColumn.className = "copy-column";
  const sourceLabel = document.createElement("div");
  sourceLabel.className = "copy-label";
  sourceLabel.textContent = t("sourceText");
  const sourceTextNode = document.createElement("p");
  sourceTextNode.className = "copy-text";
  sourceTextNode.textContent = segment.sourceText;
  sourceColumn.append(sourceLabel, sourceTextNode);

  const translationColumn = document.createElement("div");
  translationColumn.className = "copy-column translation-column";
  const translationLabel = document.createElement("div");
  translationLabel.className = "copy-label";
  translationLabel.textContent = t("translatedText");
  const translationTextNode = document.createElement("p");
  translationTextNode.className = "copy-text translation-text pending";
  translationTextNode.textContent = t("translationPending");
  const retryButton = document.createElement("button");
  retryButton.className = "btn btn-sm btn-outline-danger mt-2 d-none";
  retryButton.type = "button";
  retryButton.dataset.retrySegment = String(segment.id);
  retryButton.textContent = t("retryTranslation");
  translationColumn.append(translationLabel, translationTextNode, retryButton);

  copy.append(sourceColumn, translationColumn);
  card.append(meta, copy);
  segmentList.append(card);
  segmentNodes.set(segment.id, {
    card,
    translationTextNode,
    retryButton,
  });
  notifyParentHeight();
}

function showSegmentTranslation(segmentId, state, message = "") {
  const nodes = segmentNodes.get(segmentId);
  if (!nodes) return;
  nodes.translationTextNode.classList.remove("pending", "ready", "failed");
  nodes.translationTextNode.classList.add(state);
  nodes.translationTextNode.textContent = message || (
    state === "failed" ? t("translationFailed") : t("translationPending")
  );
  nodes.retryButton.classList.toggle("d-none", state !== "failed");
  nodes.card.classList.toggle("d-none", state !== "ready");
  if (state === "ready") {
    emptyState.classList.add("d-none");
    if (playbackSyncEnabled) {
      syncPlaybackToSubtitles();
    } else {
      scrollSegmentsToLatest();
    }
  }
  notifyParentHeight();
}

function pendingCharacterCount() {
  return pendingSegments.reduce((total, segment) => total + segment.sourceText.length, 0);
}

function scheduleBatch() {
  if (batchTimer || pendingSegments.length === 0) return;
  batchTimer = window.setTimeout(() => {
    batchTimer = null;
    flushPendingSegments();
  }, BATCH_DELAY_MS);
}

function takeNextBatch() {
  if (pendingSegments.length === 0) return [];
  const language = pendingSegments[0].language;
  const batch = [];
  let characters = 0;
  while (pendingSegments.length > 0 && batch.length < BATCH_MAX_SEGMENTS) {
    const candidate = pendingSegments[0];
    if (candidate.language !== language) break;
    const nextCharacters = characters + candidate.sourceText.length;
    if (batch.length > 0 && nextCharacters > BATCH_MAX_CHARACTERS) break;
    batch.push(pendingSegments.shift());
    characters = nextCharacters;
  }
  return batch;
}

function flushPendingSegments({ final = false } = {}) {
  if (batchTimer) {
    clearTimeout(batchTimer);
    batchTimer = null;
  }
  const remainingLookahead = final ? 0 : 1;
  while (pendingSegments.length > remainingLookahead) {
    const lookahead = final ? null : pendingSegments.pop();
    const batch = takeNextBatch();
    if (lookahead) pendingSegments.push(lookahead);
    if (batch.length === 0) break;
    enqueueTranslation(batch);
  }
  updateTranslationProgress();
}

function addPendingSegment(segment) {
  pendingSegments.push(segment);
  setTranslationActive(true);
  updateTranslationProgress();
  if (
    pendingSegments.length >= BATCH_SEGMENT_TRIGGER
    || pendingCharacterCount() >= BATCH_CHARACTER_TRIGGER
  ) {
    flushPendingSegments();
  } else {
    scheduleBatch();
  }
}

function buildContextSegments(batch) {
  const batchIds = new Set(batch.map(segment => segment.id));
  const firstBatchId = Math.min(...batchIds);
  const candidates = sortedSourceSegments()
    .filter(segment => segment.id < firstBatchId && !batchIds.has(segment.id))
    .filter(segment => translatedSegments.has(segment.id))
    .slice(-CONTEXT_MAX_SEGMENTS)
    .map(segment => ({
      id: segment.id,
      source_text: segment.sourceText,
      translated_text: translatedSegments.get(segment.id).translatedText,
    }));

  const batchCharacters = batch.reduce(
    (total, segment) => total + segment.sourceText.length
      + (segment.lowConfidenceSpans || []).reduce(
        (spanTotal, span) => spanTotal + span.length,
        0,
      ),
    0,
  );
  while (
    candidates.length > 0
    && batchCharacters + candidates.reduce(
      (total, segment) => total + segment.source_text.length + segment.translated_text.length,
      0,
    ) > REQUEST_MAX_CHARACTERS
  ) {
    candidates.shift();
  }
  return candidates;
}

function buildFollowingContextSegments(batch, contextSegments) {
  const batchIds = new Set(batch.map(segment => segment.id));
  const lastBatchId = Math.max(...batchIds);
  const candidates = sortedSourceSegments()
    .filter(segment => segment.id > lastBatchId && !batchIds.has(segment.id))
    .slice(0, CONTEXT_MAX_SEGMENTS)
    .map(segment => ({
      id: segment.id,
      text: segment.sourceText,
    }));

  const fixedCharacters = batch.reduce(
    (total, segment) => total + segment.sourceText.length
      + (segment.lowConfidenceSpans || []).reduce(
        (spanTotal, span) => spanTotal + span.length,
        0,
      ),
    0,
  ) + contextSegments.reduce(
    (total, segment) => total + segment.source_text.length + segment.translated_text.length,
    0,
  );
  while (
    candidates.length > 0
    && fixedCharacters + candidates.reduce(
      (total, segment) => total + segment.text.length,
      0,
    ) > REQUEST_MAX_CHARACTERS
  ) {
    candidates.pop();
  }
  return candidates;
}

function translationErrorMessage(data, fallback) {
  if (!data || typeof data !== "object") return fallback;
  if (typeof data.detail === "string") return data.detail;
  if (data.detail && typeof data.detail.message === "string") return data.detail.message;
  if (typeof data.message === "string") return data.message;
  return fallback;
}

function responseIsRetryable(response, data) {
  return RETRYABLE_HTTP_STATUSES.has(response.status)
    || data?.retryable === true
    || data?.detail?.retryable === true;
}

async function requestTranslation(payload) {
  let response;
  try {
    response = await fetch("/api/youtube-live/translate-batch", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Translation-Token": currentTranslationToken,
      },
      body: JSON.stringify(payload),
    });
  } catch (error) {
    const wrapped = new Error(error.message || t("translationServiceFailed"));
    wrapped.retryable = true;
    wrapped.code = "NETWORK_ERROR";
    throw wrapped;
  }

  let data = null;
  try {
    data = await response.json();
  } catch (_) {}

  if (!response.ok) {
    const error = new Error(translationErrorMessage(data, t("translationServiceFailed")));
    error.retryable = responseIsRetryable(response, data);
    error.status = response.status;
    const upstreamCode = data?.code
      ?? data?.error_code
      ?? data?.detail?.code
      ?? data?.detail?.error_code;
    if (upstreamCode !== undefined && upstreamCode !== null) {
      error.code = String(upstreamCode);
    }
    throw error;
  }
  return data;
}

function invalidTranslationResponseError() {
  const error = new Error(t("invalidTranslation"));
  error.code = "INVALID_RESPONSE";
  return error;
}

function validateTranslationResponse(payload, data, batch) {
  if (!data || typeof data !== "object" || data.request_id !== payload.request_id) {
    throw invalidTranslationResponseError();
  }
  if (!Array.isArray(data.translations) || data.translations.length !== batch.length) {
    throw invalidTranslationResponseError();
  }
  const expectedIds = batch.map(segment => segment.id);
  const actualIds = data.translations.map(result => Number(result?.id));
  if (expectedIds.some((id, index) => id !== actualIds[index])) {
    throw invalidTranslationResponseError();
  }
  for (const result of data.translations) {
    if (typeof result.translated_text !== "string" || !result.translated_text.trim()) {
      throw invalidTranslationResponseError();
    }
  }
  return data.translations;
}

async function translateBatch(batch) {
  const batchNumber = ++batchCounter;
  const requestId = `youtube-${currentJobId}-${selectedTargetLanguage}-batch-${batchNumber}-${PROMPT_VERSION}`;
  const contextSegments = buildContextSegments(batch);
  const payload = {
    request_id: requestId,
    source_language: batch[0].language,
    target_language: selectedTargetLanguage,
    prompt_version: PROMPT_VERSION,
    context_segments: contextSegments,
    following_context_segments: buildFollowingContextSegments(batch, contextSegments),
    segments: batch.map(segment => ({
      id: segment.id,
      text: segment.sourceText,
      ...(
        segment.lowConfidenceSpans.length > 0
          ? { low_confidence_spans: segment.lowConfidenceSpans }
          : {}
      ),
    })),
  };

  let lastError = null;
  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt += 1) {
    try {
      const data = await requestTranslation(payload);
      const results = validateTranslationResponse(payload, data, batch);
      for (const result of results) {
        const translatedText = result.translated_text.trim();
        translatedSegments.set(Number(result.id), {
          id: Number(result.id),
          translatedText,
        });
        failedSegmentIds.delete(Number(result.id));
        translationFailureCodes.delete(Number(result.id));
        showSegmentTranslation(Number(result.id), "ready", translatedText);
      }
      return;
    } catch (error) {
      lastError = error;
      if (!error.retryable || attempt >= RETRY_DELAYS_MS.length) break;
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAYS_MS[attempt]));
    }
  }

  const failureCode = translationErrorCode(lastError);
  for (const segment of batch) {
    failedSegmentIds.add(segment.id);
    translationFailureCodes.set(segment.id, failureCode);
    showSegmentTranslation(
      segment.id,
      "failed",
      lastError?.message || t("translationFailed"),
    );
  }
}

function enqueueTranslation(batch) {
  if (!batch.length) return translationQueue;
  queuedBatchCount += 1;
  setTranslationActive(true);
  for (const segment of batch) showSegmentTranslation(segment.id, "pending");

  translationQueue = translationQueue.then(async () => {
    try {
      await translateBatch(batch);
    } catch (error) {
      console.error("Translation queue failed", error);
      const failureCode = translationErrorCode(error);
      for (const segment of batch) {
        failedSegmentIds.add(segment.id);
        translationFailureCodes.set(segment.id, failureCode);
        showSegmentTranslation(segment.id, "failed", error.message || t("translationFailed"));
      }
    } finally {
      queuedBatchCount = Math.max(0, queuedBatchCount - 1);
      updateTranslationProgress();
      maybeFinalize();
    }
  });
  return translationQueue;
}

function effectiveSourceLanguage(dataLanguage) {
  return requestedSourceLanguage || normalizeTranslationLanguage(dataLanguage);
}

function addSegment(data) {
  const id = Number(data.index);
  const sourceText = String(data.text || "").trim();
  if (!Number.isInteger(id) || id <= 0 || !sourceText || sourceSegments.has(id)) return;

  const start = Number(data.start) || 0;
  const parsedEnd = Number(data.end);
  const end = Number.isFinite(parsedEnd) && parsedEnd > start ? parsedEnd : start + 3;
  const normalizedLanguage = effectiveSourceLanguage(data.language);
  if (!detectedSourceLanguage && normalizedLanguage) detectedSourceLanguage = normalizedLanguage;
  const segment = {
    id,
    start,
    end,
    sourceText,
    language: normalizedLanguage,
    rawLanguage: String(data.language || ""),
    lowConfidenceSpans: Array.isArray(data.low_confidence_spans)
      ? data.low_confidence_spans
        .filter(span => typeof span === "string" && sourceText.includes(span))
        .slice(0, 20)
      : [],
  };
  sourceSegments.set(id, segment);
  renderSegment(segment);
  updateTranscriptionProgress(data);

  if (!normalizedLanguage) {
    const rawLanguage = segment.rawLanguage || "unknown";
    unsupportedLanguageNotice = t("unsupportedLanguage", { language: rawLanguage });
    failedSegmentIds.add(id);
    translationFailureCodes.set(id, "UNSUPPORTED_LANGUAGE");
    showSegmentTranslation(id, "failed", unsupportedLanguageNotice);
    setStatus(appendTranslationErrorCodes(unsupportedLanguageNotice), "failed");
  } else if (normalizedLanguage === selectedTargetLanguage) {
    translatedSegments.set(id, { id, translatedText: sourceText });
    showSegmentTranslation(id, "ready", sourceText);
    if (!sameLanguageNoticeShown) {
      sameLanguageNoticeShown = true;
      setStatus(t("sameLanguage"), "running");
    }
  } else {
    addPendingSegment(segment);
  }
  updateTranslationProgress();
}

function maybeFinalize() {
  if (!transcriptionDone || pendingSegments.length > 0 || queuedBatchCount > 0) {
    setTranslationActive(pendingSegments.length > 0 || queuedBatchCount > 0);
    return;
  }
  setTranslationActive(false);
  setTranscriptionActive(false);
  updateTranslationProgress();
  updateActionButtons();
  actionBar.classList.remove("d-none");
  const hasFailures = failedSegmentIds.size > 0;
  if (transcriptionFailed) {
    setStatus(t("failed"), "failed");
  } else if (hasFailures) {
    setStatus(appendTranslationErrorCodes(t("partialDone")), "failed");
  } else {
    setStatus(t("done"), "idle");
  }
  resetCaptchaState(false);
  notifyParentHeight();
}

function updateActionButtons() {
  const partial = sourceSegments.size > translatedSegments.size;
  downloadSourceSrt.textContent = t("downloadSourceSrt");
  downloadTranslatedSrt.textContent = t("downloadTranslatedSrt") + (partial ? t("partialSuffix") : "");
  downloadBilingualSrt.textContent = t("downloadBilingualSrt") + (partial ? t("partialSuffix") : "");
  downloadSegmentsJson.textContent = t("downloadJson");
  nextTranscription.textContent = t("nextTranscription");
}

function transcriptFilename(suffix) {
  const base = (videoTitle.textContent || "youtube_transcript")
    .replace(/[\\/:*?"<>|]+/g, "_")
    .slice(0, 80)
    || "youtube_transcript";
  return `${base}_${suffix}`;
}

function buildSrt(mode) {
  return sortedSourceSegments().map((segment, index) => {
    const translation = translatedSegments.get(segment.id)?.translatedText
      || segment.sourceText;
    let text = segment.sourceText;
    if (mode === "translated") text = translation;
    if (mode === "bilingual") text = `${translation}\n${segment.sourceText}`;
    return `${index + 1}\n${formatSrtTimestamp(segment.start)} --> ${formatSrtTimestamp(segment.end)}\n${text}`;
  }).join("\n\n") + (sourceSegments.size ? "\n" : "");
}

function buildSegmentsJson() {
  return JSON.stringify({
    schema_version: 1,
    source_language: requestedSourceLanguage || detectedSourceLanguage || null,
    target_language: selectedTargetLanguage,
    segments: sortedSourceSegments().map(segment => ({
      id: segment.id,
      start_ms: Math.round(segment.start * 1000),
      end_ms: Math.round(segment.end * 1000),
      source_text: segment.sourceText,
      translated_text: translatedSegments.get(segment.id)?.translatedText
        || segment.sourceText,
    })),
  }, null, 2);
}

function downloadTextFile(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function confirmPartialDownload() {
  return sourceSegments.size === translatedSegments.size
    || window.confirm(t("partialDownloadWarning"));
}

downloadSourceSrt.addEventListener("click", () => {
  downloadTextFile(
    buildSrt("source"),
    transcriptFilename("source.srt"),
    "application/x-subrip;charset=utf-8",
  );
});

downloadTranslatedSrt.addEventListener("click", () => {
  if (!confirmPartialDownload()) return;
  downloadTextFile(
    buildSrt("translated"),
    transcriptFilename("translated.srt"),
    "application/x-subrip;charset=utf-8",
  );
});

downloadBilingualSrt.addEventListener("click", () => {
  if (!confirmPartialDownload()) return;
  downloadTextFile(
    buildSrt("bilingual"),
    transcriptFilename("bilingual.srt"),
    "application/x-subrip;charset=utf-8",
  );
});

downloadSegmentsJson.addEventListener("click", () => {
  if (!confirmPartialDownload()) return;
  downloadTextFile(
    buildSegmentsJson(),
    transcriptFilename("segments.json"),
    "application/json;charset=utf-8",
  );
});

nextTranscription.addEventListener("click", () => window.location.reload());

segmentList.addEventListener("click", event => {
  const seekButton = event.target.closest("[data-seek-time]");
  if (seekButton && youtubePlayerController?.seekTo) {
    const seekTime = Number(seekButton.dataset.seekTime);
    if (Number.isFinite(seekTime)) {
      playbackSyncEnabled = true;
      youtubePlayerController.seekTo(seekTime, true);
      focusPlaybackSegment(playbackSegmentAt(seekTime));
    }
    return;
  }

  const button = event.target.closest("[data-retry-segment]");
  if (!button) return;
  const segmentId = Number(button.dataset.retrySegment);
  const segment = sourceSegments.get(segmentId);
  if (!segment || !segment.language) return;
  failedSegmentIds.delete(segmentId);
  translationFailureCodes.delete(segmentId);
  translatedSegments.delete(segmentId);
  showSegmentTranslation(segmentId, "pending");
  setTranslationActive(true);
  actionBar.classList.add("d-none");
  enqueueTranslation([segment]);
  updateTranslationProgress();
});

async function readError(response) {
  try {
    const data = await response.json();
    return translationErrorMessage(data, t("requestFailed"));
  } catch (_) {
    return t("requestFailed");
  }
}

async function loadPublicConfig() {
  try {
    const response = await fetch("/api/public-config", { cache: "no-store" });
    if (!response.ok) return;
    const config = await response.json();
    if (config.maintenance?.enabled) {
      window.top.location.replace("/maintenance");
      return;
    }
    captchaEnabled = config.captcha_enabled !== false;
    if (!captchaEnabled) resetCaptchaState(false);
  } catch (_) {}
}

function resetCaptchaState(showCaptcha = true) {
  if (!captchaEnabled) showCaptcha = false;
  captchaId.value = "";
  captchaToken.value = "";
  captchaAnswer.value = "";
  captchaAnswer.disabled = false;
  captchaBlock.classList.toggle("d-none", !showCaptcha);
  captchaStatus.classList.add("d-none");
  startBtn.classList.toggle("d-none", showCaptcha);
  startBtn.disabled = false;
  verifyCaptcha.disabled = false;
  refreshCaptcha.disabled = false;
}

async function loadCaptcha() {
  if (!captchaEnabled) {
    resetCaptchaState(false);
    notifyParentHeight();
    return;
  }
  resetCaptchaState(true);
  captchaImage.removeAttribute("src");
  const response = await fetch("/api/captcha", { cache: "no-store" });
  if (!response.ok) throw new Error(await readError(response));
  const data = await response.json();
  captchaImage.src = data.image;
  captchaId.value = data.captcha_id;
  notifyParentHeight();
}

async function verifyCaptchaAnswer() {
  if (!captchaEnabled) {
    await beginTranscription();
    return;
  }
  if (!captchaId.value || !captchaAnswer.value.trim()) {
    setStatus(t("captchaRequired"), "failed");
    return;
  }
  verifyCaptcha.disabled = true;
  try {
    const response = await fetch("/api/captcha/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        captcha_id: captchaId.value,
        captcha_answer: captchaAnswer.value.trim(),
      }),
    });
    if (!response.ok) throw new Error(await readError(response));
    const data = await response.json();
    captchaToken.value = data.captcha_token;
    captchaBlock.classList.add("d-none");
    captchaStatus.classList.remove("d-none");
    captchaAnswer.disabled = true;
    await beginTranscription();
  } catch (error) {
    setStatus(error.message || t("captchaVerifyFailed"), "failed");
    await loadCaptcha();
  } finally {
    verifyCaptcha.disabled = Boolean(captchaToken.value);
    notifyParentHeight();
  }
}

refreshCaptcha.addEventListener("click", async () => {
  try {
    await loadCaptcha();
    setStatus(t("waiting"), "idle");
  } catch (error) {
    setStatus(error.message || t("captchaLoadFailed"), "failed");
  }
});
verifyCaptcha.addEventListener("click", verifyCaptchaAnswer);

async function beginTranscription() {
  resetView();
  requestedSourceLanguage = sourceLanguage.value;
  selectedTargetLanguage = targetLanguage.value;
  sourceLanguage.disabled = true;
  targetLanguage.disabled = true;
  youtubeUrl.disabled = true;
  ignoreSubtitles.disabled = true;
  startBtn.disabled = true;
  startBtn.classList.add("d-none");
  captchaBlock.classList.add("d-none");
  setTranscriptionActive(true);
  setStatus(t("creating"), "running");
  if (requestedSourceLanguage) revealResults();

  const whisperLanguage = requestedSourceLanguage === "zh-TW"
    ? "zh"
    : requestedSourceLanguage;

  try {
    const response = await fetch("/api/youtube-live/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: youtubeUrl.value.trim(),
        language: whisperLanguage,
        ignore_subtitles: ignoreSubtitles.checked,
        captcha_token: captchaToken.value,
      }),
    });
    if (!response.ok) throw new Error(await readError(response));
    const job = await response.json();
    if (!job.job_id || !job.events_url || !job.translation_token) {
      throw new Error(t("requestFailed"));
    }
    currentJobId = job.job_id;
    currentTranslationToken = job.translation_token;

    eventSource = new EventSource(job.events_url);
    eventSource.addEventListener("status", event => {
      const data = JSON.parse(event.data);
      setStatus(data.message || t("processing"), "running");
    });
    eventSource.addEventListener("metadata", event => {
      const data = JSON.parse(event.data);
      totalDuration = Number(data.duration) || 0;
      videoTitle.textContent = data.title || t("videoTitle");
      videoDetail.textContent = totalDuration
        ? `${t("durationPrefix")} ${Math.round(totalDuration)} ${t("seconds")}`
        : "";
      transcriptionDetail.textContent = totalDuration
        ? `${t("durationPrefix")} ${Math.round(totalDuration)} ${t("seconds")}`
        : t("progressDetail");
      videoMeta.classList.remove("d-none");
      progressGrid.classList.remove("d-none");
      notifyParentHeight();
    });
    eventSource.addEventListener("language_detected", event => {
      try {
        showLanguageConfirmation(JSON.parse(event.data || "{}"));
      } catch (error) {
        console.error("Could not process language detection", error);
      }
    });
    eventSource.addEventListener("segment", event => {
      try {
        addSegment(JSON.parse(event.data));
      } catch (error) {
        console.error("Could not process segment", error);
      }
    });
    eventSource.addEventListener("done", event => {
      transcriptionDone = true;
      setTranscriptionActive(false);
      setTranscriptionProgress(100, t("done"));
      try {
        const data = JSON.parse(event.data || "{}");
        const normalized = normalizeTranslationLanguage(data.language);
        if (!detectedSourceLanguage && normalized) detectedSourceLanguage = normalized;
      } catch (_) {}
      flushPendingSegments({ final: true });
      if (eventSource) eventSource.close();
      eventSource = null;
      maybeFinalize();
    });
    eventSource.addEventListener("failed", event => {
      transcriptionDone = true;
      transcriptionFailed = true;
      setTranscriptionActive(false);
      let message = t("failed");
      try {
        message = JSON.parse(event.data || "{}").message || message;
      } catch (_) {}
      setStatus(message, "failed");
      flushPendingSegments({ final: true });
      if (eventSource) eventSource.close();
      eventSource = null;
      maybeFinalize();
    });
    eventSource.onerror = () => {
      if (!eventSource) return;
      transcriptionDone = true;
      transcriptionFailed = true;
      setTranscriptionActive(false);
      setStatus(t("disconnected"), "failed");
      eventSource.close();
      eventSource = null;
      flushPendingSegments({ final: true });
      maybeFinalize();
    };
  } catch (error) {
    transcriptionDone = true;
    transcriptionFailed = true;
    setTranscriptionActive(false);
    setStatus(error.message || t("requestFailed"), "failed");
    sourceLanguage.disabled = false;
    targetLanguage.disabled = false;
    youtubeUrl.disabled = false;
    ignoreSubtitles.disabled = false;
    resetCaptchaState(false);
    maybeFinalize();
  }
}

confirmedSourceLanguage.addEventListener("change", () => {
  confirmLanguageBtn.disabled = !confirmedSourceLanguage.value;
});

confirmLanguageBtn.addEventListener("click", async () => {
  const language = confirmedSourceLanguage.value;
  if (!language || !currentJobId || !currentTranslationToken) return;
  confirmLanguageBtn.disabled = true;
  setStatus(t("confirmingLanguage"), "running");
  try {
    const response = await fetch(`/api/youtube-live/jobs/${encodeURIComponent(currentJobId)}/language`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Translation-Token": currentTranslationToken,
      },
      body: JSON.stringify({ language }),
    });
    if (!response.ok) throw new Error(await readError(response));
    requestedSourceLanguage = language;
    detectedSourceLanguage = language;
    languageConfirmation.classList.add("d-none");
    revealResults();
    setStatus(t("creating"), "running");
  } catch (error) {
    setStatus(error.message || t("requestFailed"), "failed");
    confirmLanguageBtn.disabled = false;
  }
});

youtubeUrl.addEventListener("input", scheduleVideoPreview);
youtubeUrl.addEventListener("change", updateVideoPreview);
sourceLanguage.addEventListener("change", updateStartButton);

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }
  updateVideoPreview();
  if (!captchaToken.value) {
    if (!captchaEnabled) {
      await beginTranscription();
      return;
    }
    startBtn.disabled = true;
    try {
      await loadCaptcha();
      setStatus(t("captchaRequired"), "idle");
    } catch (error) {
      setStatus(error.message || t("captchaLoadFailed"), "failed");
      resetCaptchaState(false);
    }
    return;
  }
  await beginTranscription();
});

applyLanguage(currentLanguage);
resetView();
resetCaptchaState(false);
setStatus(t("waiting"), "idle");
loadPublicConfig();
notifyParentHeight();
window.addEventListener("load", notifyParentHeight);
window.addEventListener("resize", notifyParentHeight);
