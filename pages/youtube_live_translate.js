"use strict";

const {
  contentSignature,
  displayCueAt,
  extractTitleTerms,
  normalizeSrtTimeline,
  sourceGroupContext,
  sourceTextForOverlay,
  translationForSourceId,
  validateDisplayCues,
  withPrecedingSourceContext,
} = window.SubtitleDisplayCues;

const form = document.getElementById("translateForm");
const topBrand = document.getElementById("topBrand");
const transcribeOnlyLink = document.getElementById("transcribeOnlyLink");
const youtubeSourceGroup = document.getElementById("youtubeSourceGroup");
const uploadSourceGroup = document.getElementById("uploadSourceGroup");
const youtubeUrl = document.getElementById("youtubeUrl");
const playlistPanel = document.getElementById("playlistPanel");
const playlistTitle = document.getElementById("playlistTitle");
const playlistCount = document.getElementById("playlistCount");
const playlistStatus = document.getElementById("playlistStatus");
const playlistItems = document.getElementById("playlistItems");
const videoFile = document.getElementById("videoFile");
const videoDropZone = document.getElementById("videoDropZone");
const uploadFilePanel = document.getElementById("uploadFilePanel");
const uploadFileCount = document.getElementById("uploadFileCount");
const uploadFileItems = document.getElementById("uploadFileItems");
const ignoreSubtitles = document.getElementById("ignoreSubtitles");
const includeWordTimestamps = document.getElementById("includeWordTimestamps");
const sourceLanguage = document.getElementById("sourceLanguage");
const targetLanguage = document.getElementById("targetLanguage");
const transcriptionMode = document.getElementById("transcriptionMode");
const startBtn = document.getElementById("startBtn");
const languageConfirmation = document.getElementById("languageConfirmation");
const detectedLanguageTitle = document.getElementById("detectedLanguageTitle");
const languageConfidence = document.getElementById("languageConfidence");
const confirmedSourceLanguage = document.getElementById("confirmedSourceLanguage");
const confirmLanguageBtn = document.getElementById("confirmLanguageBtn");
const videoPreview = document.getElementById("videoPreview");
const videoPreviewFrame = document.getElementById("videoPreviewFrame");
const videoFullscreenButton = document.getElementById("videoFullscreenButton");
const videoPlayerLayer = document.getElementById("videoPlayerLayer");
const videoSubtitleOverlay = document.getElementById("videoSubtitleOverlay");
const videoSubtitleSource = document.getElementById("videoSubtitleSource");
const videoSubtitleTranslation = document.getElementById("videoSubtitleTranslation");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const cancelJob = document.getElementById("cancelJob");
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
const uploadBatchDownloads = document.getElementById("uploadBatchDownloads");
const uploadDownloadList = document.getElementById("uploadDownloadList");
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
const BATCH_CHARACTER_TRIGGER = 1200;
const BATCH_MAX_SEGMENTS = 40;
const BATCH_MAX_CHARACTERS = 2000;
const GROUPING_MAX_CHARACTERS = 6000;
const SOURCE_GROUP_DURATION_TRIGGER_SECONDS = 30;
const SOURCE_BOUNDARY_HINT_RE = /[.!?。！？…]+["'”’»）)\]】」』]*\s*$/;
const SOURCE_WEAK_BOUNDARY_HINT_RE = /[,;:，、；：]["'”’»）)\]】」』]*\s*$/;
const SOURCE_LONG_PAUSE_SECONDS = 0.8;
const SOURCE_WEAK_PUNCTUATION_PAUSE_SECONDS = 0.3;
const TRANSLATION_WAVE_SIZE = 2;
const RETRY_DELAYS_MS = [2000, 5000];
const RETRYABLE_HTTP_STATUSES = new Set([429, 502, 503, 504]);
const GROUPING_VERSION = "source-groups-v1";
const GROUP_TRANSLATION_PROMPT_VERSION = "subtitle-groups-v2";
const NON_RETRYABLE_OUTPUT_CODES = new Set([
  "INVALID_RESPONSE",
  "INVALID_TRANSLATION_OUTPUT",
]);
const MAX_UPLOAD_VIDEO_FILES = 10;
const SUPPORTED_VIDEO_EXTENSION_RE = /\.(avi|m4v|mkv|mov|mp4|mpeg|mpg|ts|webm|wmv)$/i;

const params = new URLSearchParams(window.location.search);
const uploadMode = params.get("source") === "upload"
  || window.location.pathname === "/video-upload-translate";
const languageStorageKey = "audioTranscribeLanguage";
const translations = {
  "zh-Hant": {
    pageTitle: "Video Translate", uploadPageTitle: "影片上傳翻譯", transcribeOnly: "僅轉譯", urlLabel: "YouTube 網址", videoFileLabel: "選擇影片", dropVideos: "拖曳影片到這裡", dropVideosBrowse: "或點擊選擇影片", videoFileHelp: "一次最多選擇 10 支影片；檔案只用於本次轉譯，完成、取消或逾時後會自動清除。", invalidVideoFiles: "只支援影片檔案，已略過不支援的檔案。", videoFileLimit: "一次最多只能選擇 10 支影片。", videosSelected: "已選擇 {count} 支影片", uploadPrivacyLink: "查看隱私權政策", ignoreSubtitles: "忽略內建字幕", includeWordTimestamps: "逐字顯示原文",
    selectedVideos: "已選影片", selectedVideoCount: "{count} 支", videoBatchQueued: "排程中", videoBatchProcessing: "處理中", videoBatchDone: "已完成", videoBatchFailed: "失敗", videoBatchCancelled: "已取消", batchDownloads: "批次下載",
    playlistLabel: "播放清單", playlistCount: "{count} 支影片", playlistLoading: "正在讀取播放清單…", playlistLoadFailed: "無法讀取播放清單", selectPlaylistVideo: "選擇第 {index} 支影片：{title}",
    videoPreview: "影片預覽", enterFullscreen: "全螢幕", exitFullscreen: "結束全螢幕", subtitleWaiting: "字幕完成後會顯示在這裡",
    sourceLanguage: "原文語言", targetLanguage: "翻譯目標語言", transcriptionMode: "轉譯模式", modeAccurate: "高精度（較慢）", modeFast: "快速", autoDetect: "自動偵測", languageEnglish: "英文", languageJapanese: "日文", languageKorean: "韓文", languageThai: "泰文", languageTraditionalChinese: "繁體中文",
    start: "開始轉譯並翻譯", detectLanguage: "偵測語言", waiting: "等待輸入網址", waitingUpload: "等待選擇影片", creating: "建立任務中", uploading: "正在上傳影片", processing: "處理中", cancelJob: "取消轉譯", cancelled: "已取消轉譯與翻譯", transcriptionDone: "轉譯完成，翻譯繼續處理中", done: "轉譯與翻譯完成", partialDone: "處理完成，部分翻譯失敗", failed: "處理失敗", disconnected: "連線中斷", requestFailed: "請求失敗",
    detectedLanguageLabel: "偵測結果", detectedLanguageValue: "偵測為 {language}", confirmSourceLanguage: "確認原文語言", confirmAndStart: "確認並開始", confirmDetectedLanguage: "請確認原文語言後再開始轉譯", confirmingLanguage: "正在送出語言選擇", confidence: "偵測信心 {percent}%", subtitleLanguageSource: "來自 YouTube 字幕語言", unknownLanguage: "未知語言",
    resultTitle: "即時字幕", emptyState: "原文與譯文會一段一段顯示在這裡。", segmentUnit: "段", sourceText: "原文", translatedText: "翻譯", translationPending: "正在等待翻譯…", translationFailed: "翻譯失敗", retryTranslation: "重新翻譯",
    transcriptionProgress: "轉譯進度", translationProgress: "翻譯進度", translationSkipped: "已略過", progressDetail: "取得影片長度後會顯示轉譯進度。", translationWaiting: "等待轉譯內容。", estimatingCompletion: "正在估算完成時間…", estimatedCompletion: "預計完成時間 {time}", durationPrefix: "長度", seconds: "秒",
    translationCounts: "已翻譯 {translated} / 已收到 {received} 段 · 等待翻譯 {waiting} 段", translationPercentDone: "翻譯完成 {percent}% · 失敗 {failed} 段", translationErrorCodes: "錯誤碼 {codes}", sameLanguage: "原文與目標語言相同，已略過翻譯。", unsupportedLanguage: "目前翻譯服務不支援偵測到的語言：{language}",
    downloadSourceSrt: "下載原始 SRT", downloadTranslatedSrt: "下載翻譯 SRT", downloadBilingualSrt: "下載雙語 SRT", downloadJson: "下載 segments JSON", partialSuffix: "（部分完成）", partialDownloadWarning: "部分段落尚未翻譯，下載檔將以原文補位。是否繼續？", nextTranscription: "下一個轉譯內容",
    captchaLabel: "驗證碼", captchaPlaceholder: "輸入圖片中的文字", refreshCaptcha: "重新選擇", verifyCaptcha: "驗證", captchaVerified: "驗證完成", captchaLoadFailed: "取得驗證碼失敗", captchaVerifyFailed: "驗證失敗", captchaRequired: "請先完成驗證碼",
    translationServiceFailed: "翻譯服務暫時無法使用", invalidTranslation: "翻譯回應格式不正確", videoTitle: "YouTube 影片", uploadedVideoTitle: "上傳的影片", about: "關於我們", privacy: "隱私權政策", terms: "使用條款", contact: "聯絡我們", leaveWarning: "轉譯或翻譯仍在進行中，離開頁面將取消這次工作。"
  },
  en: {
    pageTitle: "Video Translate", uploadPageTitle: "Upload Video Translate", transcribeOnly: "Transcribe only", urlLabel: "YouTube URL", videoFileLabel: "Choose videos", dropVideos: "Drop videos here", dropVideosBrowse: "or click to choose videos", videoFileHelp: "Choose up to 10 videos. Files are used only for this batch and deleted after completion, cancellation, or expiration.", invalidVideoFiles: "Only video files are supported. Unsupported files were skipped.", videoFileLimit: "You can select up to 10 videos at a time.", videosSelected: "{count} videos selected", uploadPrivacyLink: "View Privacy Policy", ignoreSubtitles: "Ignore built-in subtitles", includeWordTimestamps: "Reveal source word by word",
    selectedVideos: "Selected videos", selectedVideoCount: "{count} videos", videoBatchQueued: "Queued", videoBatchProcessing: "Processing", videoBatchDone: "Complete", videoBatchFailed: "Failed", videoBatchCancelled: "Cancelled", batchDownloads: "Batch downloads",
    playlistLabel: "Playlist", playlistCount: "{count} videos", playlistLoading: "Loading playlist…", playlistLoadFailed: "Could not load playlist", selectPlaylistVideo: "Select video {index}: {title}",
    videoPreview: "Video preview", enterFullscreen: "Fullscreen", exitFullscreen: "Exit fullscreen", subtitleWaiting: "Subtitles will appear here when ready",
    sourceLanguage: "Source language", targetLanguage: "Target language", transcriptionMode: "Transcription mode", modeAccurate: "High accuracy (slower)", modeFast: "Fast", autoDetect: "Auto detect", languageEnglish: "English", languageJapanese: "Japanese", languageKorean: "Korean", languageThai: "Thai", languageTraditionalChinese: "Traditional Chinese",
    start: "Transcribe and translate", detectLanguage: "Detect language", waiting: "Waiting for a URL", waitingUpload: "Waiting for a video", creating: "Creating job", uploading: "Uploading video", processing: "Processing", cancelJob: "Cancel", cancelled: "Transcription and translation cancelled", transcriptionDone: "Transcription complete; translation is still running", done: "Transcription and translation complete", partialDone: "Complete with some translation failures", failed: "Processing failed", disconnected: "Connection interrupted", requestFailed: "Request failed",
    detectedLanguageLabel: "Detection result", detectedLanguageValue: "Detected as {language}", confirmSourceLanguage: "Confirm source language", confirmAndStart: "Confirm and start", confirmDetectedLanguage: "Confirm the source language to begin transcription", confirmingLanguage: "Submitting language selection", confidence: "Detection confidence {percent}%", subtitleLanguageSource: "From YouTube subtitle language", unknownLanguage: "Unknown language",
    resultTitle: "Live subtitles", emptyState: "Source text and translation will appear here segment by segment.", segmentUnit: "segments", sourceText: "Source", translatedText: "Translation", translationPending: "Waiting for translation…", translationFailed: "Translation failed", retryTranslation: "Retry",
    transcriptionProgress: "Transcription", translationProgress: "Translation", translationSkipped: "Skipped", progressDetail: "Progress appears after the video duration is available.", translationWaiting: "Waiting for transcription.", estimatingCompletion: "Estimating completion time…", estimatedCompletion: "Estimated completion {time}", durationPrefix: "Duration", seconds: "sec",
    translationCounts: "Translated {translated} / {received} received · {waiting} waiting", translationPercentDone: "Translation {percent}% · {failed} failed", translationErrorCodes: "Error code {codes}", sameLanguage: "Source and target languages match. Translation was skipped.", unsupportedLanguage: "The translation service does not support the detected language: {language}",
    downloadSourceSrt: "Download source SRT", downloadTranslatedSrt: "Download translated SRT", downloadBilingualSrt: "Download bilingual SRT", downloadJson: "Download segments JSON", partialSuffix: " (partial)", partialDownloadWarning: "Some segments are not translated. Source text will be used as fallback. Continue?", nextTranscription: "Next transcription",
    captchaLabel: "Verification", captchaPlaceholder: "Enter the text in the image", refreshCaptcha: "Choose again", verifyCaptcha: "Verify", captchaVerified: "Verified", captchaLoadFailed: "Could not load verification image", captchaVerifyFailed: "Verification failed", captchaRequired: "Please complete verification first",
    translationServiceFailed: "Translation service is temporarily unavailable", invalidTranslation: "The translation response is invalid", videoTitle: "YouTube video", uploadedVideoTitle: "Uploaded video", about: "About", privacy: "Privacy", terms: "Terms", contact: "Contact", leaveWarning: "Transcription or translation is still running. Leaving this page will cancel the job."
  },
  ja: {
    pageTitle: "Video Translate", uploadPageTitle: "動画アップロード翻訳", transcribeOnly: "文字起こしのみ", urlLabel: "YouTube URL", videoFileLabel: "動画を選択", dropVideos: "動画をここにドロップ", dropVideosBrowse: "またはクリックして選択", videoFileHelp: "一度に最大 10 本まで選択できます。完了・キャンセル・期限切れ後に自動削除されます。", invalidVideoFiles: "動画ファイルのみ対応しています。未対応のファイルは除外しました。", videoFileLimit: "一度に選択できる動画は最大 10 本です。", videosSelected: "{count} 本の動画を選択済み", uploadPrivacyLink: "プライバシーポリシーを見る", ignoreSubtitles: "内蔵字幕を無視", includeWordTimestamps: "原文を単語ごとに表示",
    selectedVideos: "選択した動画", selectedVideoCount: "{count} 本", videoBatchQueued: "待機中", videoBatchProcessing: "処理中", videoBatchDone: "完了", videoBatchFailed: "失敗", videoBatchCancelled: "キャンセル済み", batchDownloads: "一括ダウンロード",
    playlistLabel: "再生リスト", playlistCount: "{count} 本", playlistLoading: "再生リストを読み込み中…", playlistLoadFailed: "再生リストを読み込めません", selectPlaylistVideo: "{index} 本目を選択：{title}",
    videoPreview: "動画プレビュー", enterFullscreen: "全画面", exitFullscreen: "全画面を終了", subtitleWaiting: "字幕の準備ができるとここに表示されます",
    sourceLanguage: "原文の言語", targetLanguage: "翻訳先の言語", transcriptionMode: "文字起こしモード", modeAccurate: "高精度（低速）", modeFast: "高速", autoDetect: "自動検出", languageEnglish: "英語", languageJapanese: "日本語", languageKorean: "韓国語", languageThai: "タイ語", languageTraditionalChinese: "繁体字中国語",
    start: "文字起こしと翻訳を開始", detectLanguage: "言語を検出", waiting: "URL を入力してください", waitingUpload: "動画を選択してください", creating: "ジョブを作成中", uploading: "動画をアップロード中", processing: "処理中", cancelJob: "キャンセル", cancelled: "文字起こしと翻訳をキャンセルしました", transcriptionDone: "文字起こしが完了し、翻訳を続行しています", done: "文字起こしと翻訳が完了しました", partialDone: "一部の翻訳に失敗しました", failed: "処理に失敗しました", disconnected: "接続が切断されました", requestFailed: "リクエストに失敗しました",
    detectedLanguageLabel: "検出結果", detectedLanguageValue: "{language} として検出", confirmSourceLanguage: "原文の言語を確認", confirmAndStart: "確認して開始", confirmDetectedLanguage: "原文の言語を確認してから開始してください", confirmingLanguage: "言語設定を送信中", confidence: "検出の信頼度 {percent}%", subtitleLanguageSource: "YouTube 字幕の言語", unknownLanguage: "不明な言語",
    resultTitle: "リアルタイム字幕", emptyState: "原文と翻訳が順番に表示されます。", segmentUnit: "件", sourceText: "原文", translatedText: "翻訳", translationPending: "翻訳待ち…", translationFailed: "翻訳に失敗しました", retryTranslation: "再翻訳",
    transcriptionProgress: "文字起こしの進捗", translationProgress: "翻訳の進捗", translationSkipped: "省略", progressDetail: "動画の長さを取得後、進捗が表示されます。", translationWaiting: "文字起こしを待っています。", estimatingCompletion: "完了時刻を計算中…", estimatedCompletion: "完了予定 {time}", durationPrefix: "長さ", seconds: "秒",
    translationCounts: "翻訳済み {translated} / 受信 {received} 件・待機 {waiting} 件", translationPercentDone: "翻訳 {percent}%・失敗 {failed} 件", translationErrorCodes: "エラーコード {codes}", sameLanguage: "原文と翻訳先が同じため、翻訳を省略しました。", unsupportedLanguage: "検出された言語は現在サポートされていません：{language}",
    downloadSourceSrt: "原文 SRT をダウンロード", downloadTranslatedSrt: "翻訳 SRT をダウンロード", downloadBilingualSrt: "二言語 SRT をダウンロード", downloadJson: "segments JSON をダウンロード", partialSuffix: "（一部完了）", partialDownloadWarning: "未翻訳の区間は原文で補完されます。続行しますか？", nextTranscription: "次の文字起こし",
    captchaLabel: "認証コード", captchaPlaceholder: "画像内の文字を入力", refreshCaptcha: "選び直す", verifyCaptcha: "認証", captchaVerified: "認証完了", captchaLoadFailed: "認証画像を取得できませんでした", captchaVerifyFailed: "認証に失敗しました", captchaRequired: "先に認証を完了してください",
    translationServiceFailed: "翻訳サービスを利用できません", invalidTranslation: "翻訳レスポンスが不正です", videoTitle: "YouTube 動画", uploadedVideoTitle: "アップロード動画", about: "私たちについて", privacy: "プライバシー", terms: "利用規約", contact: "お問い合わせ", leaveWarning: "文字起こしまたは翻訳が進行中です。ページを離れると、この処理はキャンセルされます。"
  },
  ko: {
    pageTitle: "Video Translate", uploadPageTitle: "동영상 업로드 번역", transcribeOnly: "전사만", urlLabel: "YouTube URL", videoFileLabel: "동영상 선택", dropVideos: "동영상을 여기에 놓으세요", dropVideosBrowse: "또는 클릭하여 선택", videoFileHelp: "한 번에 최대 10개를 선택할 수 있으며 완료, 취소 또는 만료 후 자동 삭제됩니다.", invalidVideoFiles: "동영상 파일만 지원합니다. 지원하지 않는 파일은 제외했습니다.", videoFileLimit: "한 번에 최대 10개의 동영상만 선택할 수 있습니다.", videosSelected: "동영상 {count}개 선택됨", uploadPrivacyLink: "개인정보 처리방침 보기", ignoreSubtitles: "내장 자막 무시", includeWordTimestamps: "원문을 단어별로 표시",
    selectedVideos: "선택한 동영상", selectedVideoCount: "동영상 {count}개", videoBatchQueued: "대기 중", videoBatchProcessing: "처리 중", videoBatchDone: "완료", videoBatchFailed: "실패", videoBatchCancelled: "취소됨", batchDownloads: "일괄 다운로드",
    playlistLabel: "재생목록", playlistCount: "동영상 {count}개", playlistLoading: "재생목록을 불러오는 중…", playlistLoadFailed: "재생목록을 불러올 수 없습니다", selectPlaylistVideo: "{index}번 동영상 선택: {title}",
    videoPreview: "동영상 미리보기", enterFullscreen: "전체 화면", exitFullscreen: "전체 화면 종료", subtitleWaiting: "자막이 준비되면 여기에 표시됩니다",
    sourceLanguage: "원문 언어", targetLanguage: "번역 언어", transcriptionMode: "전사 모드", modeAccurate: "고정밀(느림)", modeFast: "빠름", autoDetect: "자동 감지", languageEnglish: "영어", languageJapanese: "일본어", languageKorean: "한국어", languageThai: "태국어", languageTraditionalChinese: "번체 중국어",
    start: "전사 및 번역 시작", detectLanguage: "언어 감지", waiting: "URL 입력 대기 중", waitingUpload: "동영상 선택 대기 중", creating: "작업 생성 중", uploading: "동영상 업로드 중", processing: "처리 중", cancelJob: "전사 취소", cancelled: "전사와 번역이 취소되었습니다", transcriptionDone: "전사가 완료되어 번역을 계속 처리하고 있습니다", done: "전사 및 번역 완료", partialDone: "일부 번역 실패와 함께 완료", failed: "처리 실패", disconnected: "연결이 끊겼습니다", requestFailed: "요청 실패",
    detectedLanguageLabel: "감지 결과", detectedLanguageValue: "{language}(으)로 감지", confirmSourceLanguage: "원문 언어 확인", confirmAndStart: "확인 후 시작", confirmDetectedLanguage: "원문 언어를 확인한 뒤 전사를 시작하세요", confirmingLanguage: "언어 선택을 전송하는 중", confidence: "감지 신뢰도 {percent}%", subtitleLanguageSource: "YouTube 자막 언어", unknownLanguage: "알 수 없는 언어",
    resultTitle: "실시간 자막", emptyState: "원문과 번역이 구간별로 표시됩니다.", segmentUnit: "개", sourceText: "원문", translatedText: "번역", translationPending: "번역 대기 중…", translationFailed: "번역 실패", retryTranslation: "다시 번역",
    transcriptionProgress: "전사 진행률", translationProgress: "번역 진행률", translationSkipped: "건너뜀", progressDetail: "영상 길이를 가져오면 진행률이 표시됩니다.", translationWaiting: "전사 내용을 기다리는 중입니다.", estimatingCompletion: "완료 시간을 계산하는 중…", estimatedCompletion: "예상 완료 시간 {time}", durationPrefix: "길이", seconds: "초",
    translationCounts: "번역 {translated} / 수신 {received}개 · 대기 {waiting}개", translationPercentDone: "번역 {percent}% · 실패 {failed}개", translationErrorCodes: "오류 코드 {codes}", sameLanguage: "원문과 대상 언어가 같아 번역을 건너뛰었습니다.", unsupportedLanguage: "감지된 언어는 현재 지원되지 않습니다: {language}",
    downloadSourceSrt: "원문 SRT 다운로드", downloadTranslatedSrt: "번역 SRT 다운로드", downloadBilingualSrt: "이중 언어 SRT 다운로드", downloadJson: "segments JSON 다운로드", partialSuffix: " (일부 완료)", partialDownloadWarning: "번역되지 않은 구간은 원문으로 대체됩니다. 계속하시겠습니까?", nextTranscription: "다음 전사",
    captchaLabel: "인증 코드", captchaPlaceholder: "이미지의 문자를 입력하세요", refreshCaptcha: "다시 선택", verifyCaptcha: "인증", captchaVerified: "인증 완료", captchaLoadFailed: "인증 이미지를 불러오지 못했습니다", captchaVerifyFailed: "인증에 실패했습니다", captchaRequired: "먼저 인증을 완료해 주세요",
    translationServiceFailed: "번역 서비스를 일시적으로 사용할 수 없습니다", invalidTranslation: "번역 응답이 올바르지 않습니다", videoTitle: "YouTube 영상", uploadedVideoTitle: "업로드한 동영상", about: "소개", privacy: "개인정보 처리방침", terms: "이용약관", contact: "문의", leaveWarning: "전사 또는 번역이 진행 중입니다. 페이지를 떠나면 이 작업이 취소됩니다."
  }
};

let currentLanguage = translations[params.get("lang")]
  ? params.get("lang")
  : localStorage.getItem(languageStorageKey) || "zh-Hant";
if (!translations[currentLanguage]) currentLanguage = "zh-Hant";
if (params.get("embedded") === "1") document.body.classList.add("embedded");

const sourceSegments = new Map();
const playbackSegments = [];
const translatedSegments = new Map();
const translatedGroups = new Map();
const translatedDisplayCues = [];
const sourceTranslationGroups = new Map();
const segmentNodes = new Map();
const failedSegmentIds = new Set();
const translationFailureCodes = new Map();
let pendingSegments = [];
let lastSourceGroupContext = null;
let eventSource = null;
let translationQueue = Promise.resolve();
let pendingTranslationBatches = [];
let translationWaveRunning = false;
let batchTimer = null;
let queuedBatchCount = 0;
let batchCounter = 0;
let groupingCounter = 0;
let groupingRunning = false;
let groupingPromise = Promise.resolve();
let pendingRevision = 0;
let lastGroupedRevision = -1;
let finalGroupingRequested = false;
let currentJobId = "";
let currentTranslationToken = "";
let currentCancelRequest = null;
const translationAbortControllers = new Set();
let jobCancellationRequested = false;
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
const playlistPreviewCache = new Map();
let playlistPreviewController = null;
let activePlaylistId = "";
let activePlaylistData = null;
let youtubePlayerApiPromise = null;
let youtubePlayerController = null;
let uploadPreviewUrl = "";
let selectedUploadFileIndex = 0;
const uploadSelectorUrls = [];
const uploadFileStatusNodes = new Map();
const uploadFileStates = new Map();
let uploadBatchJobs = [];
let uploadBatchJobIndex = -1;
let uploadBatchAdvanceScheduled = false;
const uploadBatchCancelRequests = new Map();
const uploadBatchResults = new Map();
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
  const pageTitle = uploadMode ? t("uploadPageTitle") : t("pageTitle");
  document.title = pageTitle;
  document.querySelectorAll("[data-i18n]").forEach(element => {
    element.textContent = t(element.dataset.i18n);
  });
  topBrand.querySelector("[data-i18n='pageTitle']").textContent = pageTitle;
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
  updateFullscreenButton();
  if (uploadMode) {
    videoFile.setAttribute("aria-label", t("dropVideosBrowse"));
    uploadFileCount.textContent = videoFile.files.length
      ? t("selectedVideoCount", { count: videoFile.files.length })
      : "";
    for (const [index, node] of uploadFileStatusNodes.entries()) {
      const state = uploadFileStates.get(index);
      if (state) node.textContent = t(state.key);
    }
  }
  if (activePlaylistData) {
    renderPlaylistPreview(activePlaylistData, parseYouTubeVideo(youtubeUrl.value)?.videoId || "");
  }
}

function configureSourceMode() {
  youtubeSourceGroup.classList.toggle("d-none", uploadMode);
  uploadSourceGroup.classList.toggle("d-none", !uploadMode);
  youtubeUrl.required = !uploadMode;
  youtubeUrl.disabled = uploadMode;
  videoFile.required = false;
  videoFile.disabled = !uploadMode;
  videoDropZone.classList.toggle("disabled", !uploadMode);
  ignoreSubtitles.disabled = uploadMode;
  transcribeOnlyLink.classList.toggle("d-none", uploadMode);
  if (uploadMode) {
    topBrand.href = "/video-upload-translate?source=upload";
  }
}

function setSourceControlsDisabled(disabled) {
  if (uploadMode) {
    videoFile.disabled = disabled;
    videoDropZone.classList.toggle("disabled", disabled);
    uploadFileItems.querySelectorAll("button").forEach(button => {
      button.disabled = disabled;
    });
  } else {
    youtubeUrl.disabled = disabled;
    ignoreSubtitles.disabled = disabled;
    playlistItems.querySelectorAll("button").forEach(button => {
      button.disabled = disabled;
    });
  }
  includeWordTimestamps.disabled = disabled;
  transcriptionMode.disabled = disabled;
}

function fullscreenElement() {
  return document.fullscreenElement || document.webkitFullscreenElement || null;
}

function updateFullscreenButton() {
  const active = fullscreenElement() === videoPreviewFrame;
  const label = t(active ? "exitFullscreen" : "enterFullscreen");
  videoFullscreenButton.textContent = label;
  videoFullscreenButton.setAttribute("aria-label", label);
  videoFullscreenButton.title = label;
}

async function toggleVideoFullscreen() {
  try {
    if (fullscreenElement() === videoPreviewFrame) {
      const exitFullscreen = document.exitFullscreen || document.webkitExitFullscreen;
      if (exitFullscreen) await exitFullscreen.call(document);
    } else {
      const requestFullscreen = videoPreviewFrame.requestFullscreen
        || videoPreviewFrame.webkitRequestFullscreen;
      if (requestFullscreen) await requestFullscreen.call(videoPreviewFrame);
    }
  } catch (error) {
    console.error("Could not toggle video fullscreen", error);
  }
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

  const rawPlaylistId = url.searchParams.get("list") || "";
  const playlistId = /^[A-Za-z0-9_-]{2,100}$/.test(rawPlaylistId)
    ? rawPlaylistId
    : "";

  const hashParams = new URLSearchParams(url.hash.replace(/^#/, ""));
  const start = parseYouTubeStartTime(
    url.searchParams.get("t")
      || url.searchParams.get("start")
      || hashParams.get("t"),
  );
  return { videoId, start, playlistId };
}

function formatPlaylistDuration(value) {
  const seconds = Math.max(0, Math.round(Number(value) || 0));
  if (!seconds) return "";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function clearPlaylistPreview() {
  activePlaylistId = "";
  activePlaylistData = null;
  if (playlistPreviewController) playlistPreviewController.abort();
  playlistPreviewController = null;
  playlistPanel.classList.add("d-none");
  playlistItems.replaceChildren();
  playlistItems.classList.add("d-none");
  playlistStatus.classList.remove("d-none");
}

function selectPlaylistVideo(item, index, playlistId) {
  const selectedUrl = new URL("https://www.youtube.com/watch");
  selectedUrl.searchParams.set("v", item.id);
  selectedUrl.searchParams.set("list", playlistId);
  selectedUrl.searchParams.set("index", String(index + 1));
  youtubeUrl.value = selectedUrl.toString();
  currentPreviewKey = "";
  updateVideoPreview();
  requestAnimationFrame(() => {
    playlistItems.querySelector(".playlist-video.active")?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
      inline: "center",
    });
  });
}

function renderPlaylistPreview(data, selectedVideoId) {
  activePlaylistData = data;
  playlistTitle.textContent = data.title || t("playlistLabel");
  playlistCount.textContent = t("playlistCount", {
    count: Number(data.total_items) || data.items.length,
  });
  playlistStatus.classList.add("d-none");
  playlistItems.classList.remove("d-none");
  playlistItems.replaceChildren();

  data.items.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "playlist-video";
    button.disabled = youtubeUrl.disabled;
    button.classList.toggle("active", item.id === selectedVideoId);
    button.setAttribute("aria-pressed", item.id === selectedVideoId ? "true" : "false");
    button.setAttribute("aria-label", t("selectPlaylistVideo", {
      index: index + 1,
      title: item.title,
    }));

    const thumbnail = document.createElement("div");
    thumbnail.className = "playlist-thumbnail";
    const image = document.createElement("img");
    image.src = item.thumbnail;
    image.alt = "";
    image.loading = "lazy";
    thumbnail.append(image);
    const duration = formatPlaylistDuration(item.duration);
    if (duration) {
      const durationBadge = document.createElement("span");
      durationBadge.className = "playlist-duration";
      durationBadge.textContent = duration;
      thumbnail.append(durationBadge);
    }

    const copy = document.createElement("div");
    copy.className = "playlist-video-copy";
    const itemIndex = document.createElement("span");
    itemIndex.className = "playlist-index";
    itemIndex.textContent = String(index + 1);
    const title = document.createElement("span");
    title.className = "playlist-video-title";
    title.textContent = item.title;
    copy.append(itemIndex, title);
    button.append(thumbnail, copy);
    button.addEventListener("click", () => selectPlaylistVideo(item, index, data.id));
    playlistItems.append(button);
  });
}

async function loadPlaylistPreview(parsed) {
  if (!parsed?.playlistId) {
    clearPlaylistPreview();
    return;
  }

  const playlistId = parsed.playlistId;
  activePlaylistId = playlistId;
  playlistPanel.classList.remove("d-none");
  const cached = playlistPreviewCache.get(playlistId);
  if (cached) {
    renderPlaylistPreview(cached, parsed.videoId);
    notifyParentHeight();
    return;
  }

  if (playlistPreviewController) playlistPreviewController.abort();
  playlistPreviewController = new AbortController();
  playlistTitle.textContent = t("playlistLabel");
  playlistCount.textContent = "";
  playlistItems.classList.add("d-none");
  playlistStatus.textContent = t("playlistLoading");
  playlistStatus.classList.remove("d-none");
  try {
    const response = await fetch(
      `/api/youtube-live/playlists/preview?url=${encodeURIComponent(youtubeUrl.value.trim())}`,
      { signal: playlistPreviewController.signal },
    );
    if (!response.ok) throw new Error(await readError(response));
    const data = await response.json();
    if (!Array.isArray(data.items) || !data.items.length) {
      throw new Error(t("playlistLoadFailed"));
    }
    playlistPreviewCache.set(playlistId, data);
    if (activePlaylistId !== playlistId) return;
    renderPlaylistPreview(data, parseYouTubeVideo(youtubeUrl.value)?.videoId || parsed.videoId);
  } catch (error) {
    if (error.name === "AbortError" || activePlaylistId !== playlistId) return;
    playlistItems.classList.add("d-none");
    playlistStatus.textContent = error.message || t("playlistLoadFailed");
    playlistStatus.classList.remove("d-none");
  } finally {
    if (activePlaylistId === playlistId) playlistPreviewController = null;
    notifyParentHeight();
  }
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

function updateVideoSubtitleOverlay(segment, currentTime = null) {
  const failureCode = segment
    ? translationFailureCodes.get(segment.id)
    : "";
  if (segment && failureCode) {
    const sourceText = sourceTextForOverlay({
      activeCue: null,
      segment,
      currentTime,
      revealWords: includeWordTimestamps.checked,
    });
    videoSubtitleSource.textContent = sourceText;
    videoSubtitleSource.classList.remove("d-none");
    videoSubtitleTranslation.textContent = `${t("translationFailed")} · ${t(
      "translationErrorCodes",
      { codes: failureCode },
    )}`;
    videoSubtitleOverlay.classList.remove("waiting");
    videoSubtitleOverlay.classList.add("error");
    return;
  }

  const activeCue = Number.isFinite(currentTime)
    ? displayCueAt(translatedDisplayCues, currentTime)
    : null;
  const translatedText = activeCue
    ? activeCue.lines.join("\n")
    : segment
      ? translatedSegments.get(segment.id)?.translatedText
      : "";
  if (!segment) {
    clearVideoSubtitleOverlay(false);
    return;
  }
  if (!translatedText) {
    const sourceText = sourceTextForOverlay({
      activeCue: null,
      segment,
      currentTime,
      revealWords: includeWordTimestamps.checked,
    });
    videoSubtitleSource.textContent = "";
    videoSubtitleSource.classList.add("d-none");
    videoSubtitleTranslation.textContent = sourceText;
    videoSubtitleOverlay.classList.remove("waiting");
    videoSubtitleOverlay.classList.remove("error");
    return;
  }

  const sourceText = sourceTextForOverlay({
    activeCue,
    segment,
    currentTime,
    revealWords: includeWordTimestamps.checked,
  });
  if (translationLanguagesMatch(segment.language, selectedTargetLanguage)) {
    videoSubtitleSource.textContent = "";
    videoSubtitleSource.classList.add("d-none");
    videoSubtitleTranslation.textContent = sourceText;
    videoSubtitleOverlay.classList.remove("waiting");
    videoSubtitleOverlay.classList.remove("error");
    return;
  }
  videoSubtitleSource.textContent = sourceText;
  videoSubtitleTranslation.textContent = translatedText;
  videoSubtitleSource.classList.toggle(
    "d-none",
    sourceText.trim() === translatedText.trim(),
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
  if (uploadPreviewUrl) {
    URL.revokeObjectURL(uploadPreviewUrl);
    uploadPreviewUrl = "";
  }
  videoPlayerLayer.replaceChildren();
  const mount = document.createElement("div");
  mount.id = "youtubePlayer";
  mount.setAttribute("title", t("videoPreview"));
  videoPlayerLayer.append(mount);
  return mount;
}

function playbackSegmentAt(seconds) {
  let lower = 0;
  let upper = playbackSegments.length - 1;
  let candidate = null;
  while (lower <= upper) {
    const middle = Math.floor((lower + upper) / 2);
    const segment = playbackSegments[middle];
    if (segment.start <= seconds) {
      candidate = segment;
      lower = middle + 1;
    } else {
      upper = middle - 1;
    }
  }
  if (
    candidate
    && seconds < candidate.end
  ) {
    return candidate;
  }
  return null;
}

function focusPlaybackSegment(segment, currentTime = null) {
  updateVideoSubtitleOverlay(segment, currentTime);
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
    const currentTime = youtubePlayerController.getCurrentTime();
    focusPlaybackSegment(playbackSegmentAt(currentTime), currentTime);
  } catch (_) {}
}

function startPlaybackSync() {
  if (playbackSyncTimer) clearInterval(playbackSyncTimer);
  playbackSyncTimer = setInterval(
    syncPlaybackToSubtitles,
    includeWordTimestamps.checked ? 100 : 400,
  );
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
      fs: 0,
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

function clearUploadSelectorUrls() {
  while (uploadSelectorUrls.length) URL.revokeObjectURL(uploadSelectorUrls.pop());
}

function isSupportedVideoFile(file) {
  return file instanceof File && (
    String(file.type || "").toLowerCase().startsWith("video/")
    || SUPPORTED_VIDEO_EXTENSION_RE.test(file.name)
  );
}

function videoFileIdentity(file) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

function dragIncludesFiles(event) {
  return Array.from(event.dataTransfer?.types || []).includes("Files");
}

function setSelectedVideoFiles(incomingFiles, { append = false } = {}) {
  const incoming = [...incomingFiles];
  const supported = incoming.filter(isSupportedVideoFile);
  const invalidCount = incoming.length - supported.length;
  const combined = append ? [...videoFile.files, ...supported] : supported;
  const unique = [];
  const identities = new Set();
  for (const file of combined) {
    const identity = videoFileIdentity(file);
    if (identities.has(identity)) continue;
    identities.add(identity);
    unique.push(file);
  }
  const limitExceeded = unique.length > MAX_UPLOAD_VIDEO_FILES;
  const selected = unique.slice(0, MAX_UPLOAD_VIDEO_FILES);
  const transfer = new DataTransfer();
  selected.forEach(file => transfer.items.add(file));
  videoFile.files = transfer.files;

  selectedUploadFileIndex = 0;
  uploadFileStates.clear();
  uploadBatchResults.clear();
  uploadDownloadList.replaceChildren();
  uploadBatchDownloads.classList.add("d-none");
  renderUploadFileSelector();
  updateVideoPreview();

  if (limitExceeded) {
    setStatus(t("videoFileLimit"), "failed");
  } else if (invalidCount > 0) {
    setStatus(t("invalidVideoFiles"), "failed");
  } else if (selected.length > 0) {
    setStatus(t("videosSelected", { count: selected.length }), "idle");
  } else {
    setStatus(t("waitingUpload"), "idle");
  }
}

function setUploadFileStatus(index, key, state = "") {
  uploadFileStates.set(index, { key, state });
  const node = uploadFileStatusNodes.get(index);
  if (!node) return;
  node.textContent = t(key);
  node.classList.remove("d-none");
  node.dataset.state = state;
}

function selectUploadFile(index) {
  const files = [...videoFile.files];
  if (!files[index]) return;
  selectedUploadFileIndex = index;
  uploadFileItems.querySelectorAll(".playlist-video").forEach((button, buttonIndex) => {
    const active = buttonIndex === index;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  currentPreviewKey = "";
  updateVideoPreview();
}

function renderUploadFileSelector() {
  if (!uploadMode) return;
  const files = [...videoFile.files];
  clearUploadSelectorUrls();
  uploadFileStatusNodes.clear();
  uploadFileItems.replaceChildren();
  uploadFilePanel.classList.toggle("d-none", files.length === 0);
  uploadFileCount.textContent = files.length
    ? t("selectedVideoCount", { count: files.length })
    : "";
  if (!files.length) return;
  if (selectedUploadFileIndex >= files.length) selectedUploadFileIndex = 0;

  files.forEach((file, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "playlist-video";
    button.disabled = videoFile.disabled;
    const active = index === selectedUploadFileIndex;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
    button.setAttribute("aria-label", file.name);

    const thumbnail = document.createElement("div");
    thumbnail.className = "upload-file-thumbnail";
    const previewVideo = document.createElement("video");
    previewVideo.muted = true;
    previewVideo.playsInline = true;
    previewVideo.preload = "metadata";
    const objectUrl = URL.createObjectURL(file);
    uploadSelectorUrls.push(objectUrl);
    previewVideo.src = objectUrl;
    previewVideo.addEventListener("loadedmetadata", () => {
      if (Number.isFinite(previewVideo.duration) && previewVideo.duration > 0.2) {
        previewVideo.currentTime = Math.min(0.2, previewVideo.duration / 4);
      }
    }, { once: true });
    thumbnail.append(previewVideo);

    const status = document.createElement("span");
    status.className = "upload-file-status d-none";
    uploadFileStatusNodes.set(index, status);
    const savedState = uploadFileStates.get(index);
    if (savedState) {
      status.textContent = t(savedState.key);
      status.classList.remove("d-none");
      status.dataset.state = savedState.state;
    }
    thumbnail.append(status);

    const copy = document.createElement("div");
    copy.className = "playlist-video-copy";
    const itemIndex = document.createElement("span");
    itemIndex.className = "playlist-index";
    itemIndex.textContent = String(index + 1);
    const title = document.createElement("span");
    title.className = "playlist-video-title";
    title.textContent = file.name;
    copy.append(itemIndex, title);
    button.append(thumbnail, copy);
    button.addEventListener("click", () => selectUploadFile(index));
    uploadFileItems.append(button);
  });
  notifyParentHeight();
}

function renderUploadedVideo(file, previewKey) {
  const mount = resetVideoPlayerMount();
  if (previewKey !== currentPreviewKey || !mount.isConnected) return;

  uploadPreviewUrl = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.controls = true;
  if (video.controlsList) video.controlsList.add("nofullscreen");
  else video.setAttribute("controlslist", "nofullscreen");
  video.playsInline = true;
  video.preload = "metadata";
  video.src = uploadPreviewUrl;
  video.setAttribute("title", t("videoPreview"));
  mount.append(video);
  youtubePlayerController = {
    getCurrentTime: () => video.currentTime,
    getDuration: () => video.duration,
    seekTo: seconds => {
      video.currentTime = Math.max(0, Number(seconds) || 0);
      playbackSyncEnabled = true;
      syncPlaybackToSubtitles();
    },
    destroy: () => {
      video.pause();
      video.removeAttribute("src");
      video.load();
    },
  };
  video.addEventListener("loadedmetadata", () => {
    if (previewKey !== currentPreviewKey) return;
    startPlaybackSync();
    notifyParentHeight();
  });
  video.addEventListener("play", () => {
    playbackSyncEnabled = true;
    syncPlaybackToSubtitles();
  });
  video.addEventListener("seeking", () => {
    playbackSyncEnabled = true;
    syncPlaybackToSubtitles();
  });
  video.addEventListener("dblclick", event => {
    event.preventDefault();
    void toggleVideoFullscreen();
  });
}

function updateVideoPreview() {
  if (uploadMode) {
    clearPlaylistPreview();
    const file = videoFile.files?.[selectedUploadFileIndex];
    if (!file) {
      currentPreviewKey = "";
      resetVideoPlayerMount();
      videoPreview.classList.add("d-none");
      notifyParentHeight();
      return false;
    }
    const previewKey = `${file.name}:${file.size}:${file.lastModified}`;
    if (previewKey !== currentPreviewKey) {
      currentPreviewKey = previewKey;
      renderUploadedVideo(file, previewKey);
    }
    videoPreview.classList.remove("d-none");
    notifyParentHeight();
    return true;
  }

  const parsed = parseYouTubeVideo(youtubeUrl.value);
  if (!parsed) {
    clearPlaylistPreview();
    currentPreviewKey = "";
    resetVideoPlayerMount();
    videoPreview.classList.add("d-none");
    notifyParentHeight();
    return false;
  }

  void loadPlaylistPreview(parsed);

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

function translationLanguagesMatch(source, target) {
  const normalizedSource = normalizeTranslationLanguage(source);
  const normalizedTarget = normalizeTranslationLanguage(target);
  return Boolean(normalizedSource && normalizedTarget && normalizedSource === normalizedTarget);
}

function currentTranslationIsSkipped() {
  return translationLanguagesMatch(
    requestedSourceLanguage || detectedSourceLanguage,
    selectedTargetLanguage,
  );
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

function sendCancellationBeacon(cancelRequest) {
  if (!cancelRequest?.url || !cancelRequest?.token) return;
  const body = JSON.stringify({ cancel_token: cancelRequest.token });
  if (navigator.sendBeacon) {
    const payload = new Blob([body], { type: "application/json" });
    if (navigator.sendBeacon(cancelRequest.url, payload)) return;
  }
  fetch(cancelRequest.url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    keepalive: true,
  }).catch(() => {});
}

function cancelCurrentJob() {
  if (!currentCancelRequest) return;
  sendCancellationBeacon(currentCancelRequest);
  currentCancelRequest = null;
}

window.addEventListener("pagehide", () => {
  if (!transcriptionActive && !translationActive) return;
  if (uploadMode && uploadBatchCancelRequests.size) {
    for (const cancelRequest of uploadBatchCancelRequests.values()) {
      sendCancellationBeacon(cancelRequest);
    }
  } else {
    cancelCurrentJob();
  }
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

function sortedDisplayCues() {
  return [...translatedDisplayCues].sort((left, right) => (
    left.start - right.start || left.end - right.end || left.cueId.localeCompare(right.cueId)
  ));
}

function sourceTextForDisplayCue(cue) {
  if (typeof cue.sourceText === "string" && cue.sourceText.trim()) {
    return cue.sourceText.trim();
  }
  const assigned = cue.sourceIds
    .map(sourceId => sourceSegments.get(sourceId)?.sourceText || "")
    .filter(Boolean);
  if (assigned.length > 0) return assigned.join(" ");
  return sortedSourceSegments()
    .filter(segment => segment.end > cue.start && segment.start < cue.end)
    .map(segment => segment.sourceText)
    .join(" ");
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
  playbackSegments.length = 0;
  translatedSegments.clear();
  translatedGroups.clear();
  translatedDisplayCues.length = 0;
  sourceTranslationGroups.clear();
  lastSourceGroupContext = null;
  segmentNodes.clear();
  failedSegmentIds.clear();
  translationFailureCodes.clear();
  pendingSegments = [];
  translationQueue = Promise.resolve();
  pendingTranslationBatches = [];
  translationWaveRunning = false;
  queuedBatchCount = 0;
  batchCounter = 0;
  groupingCounter = 0;
  groupingRunning = false;
  groupingPromise = Promise.resolve();
  pendingRevision = 0;
  lastGroupedRevision = -1;
  finalGroupingRequested = false;
  currentJobId = "";
  currentTranslationToken = "";
  currentCancelRequest = null;
  abortTranslationRequests();
  jobCancellationRequested = false;
  cancelJob.classList.add("d-none");
  cancelJob.disabled = false;
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
  transcriptionBar.classList.add("progress-bar-animated");
  translationBar.classList.add("progress-bar-animated");
  updateTranslationProgress();
  notifyParentHeight();
}

function setTranscriptionProgress(percent, detail = null) {
  const value = Math.max(0, Math.min(100, Number(percent) || 0));
  const label = value > 0 && value < 1 ? value.toFixed(1) : String(Math.round(value));
  transcriptionPercent.textContent = `${label}%`;
  transcriptionBar.style.width = `${value}%`;
  transcriptionBar.parentElement.setAttribute("aria-valuenow", String(Math.round(value)));
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
  const translationProgressElement = translationBar.parentElement;

  if (currentTranslationIsSkipped()) {
    translationPercent.textContent = t("translationSkipped");
    translationDetail.textContent = t("sameLanguage");
    translationProgressElement.classList.add("d-none");
    translationProgressElement.setAttribute("aria-valuenow", "0");
    translationBar.style.width = "0%";
    translationBar.classList.remove("progress-bar-animated");
    segmentCount.textContent = `${received} ${t("segmentUnit")}`;
    return;
  }

  translationProgressElement.classList.remove("d-none");

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
  segmentCount.textContent = `${received} ${t("segmentUnit")}`;
}

function renderSegment(segment) {
  const card = document.createElement("article");
  card.className = "segment-card";
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
  const groupBadge = document.createElement("span");
  groupBadge.className = "badge text-bg-secondary d-none";
  meta.append(time, number, groupBadge);

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
  emptyState.classList.add("d-none");
  segmentNodes.set(segment.id, {
    card,
    groupBadge,
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
  if (state === "ready") {
    if (playbackSyncEnabled) {
      syncPlaybackToSubtitles();
    }
  }
  notifyParentHeight();
}

function showSegmentTranslationGroup(segmentId, group) {
  const nodes = segmentNodes.get(segmentId);
  if (!nodes) return;
  const firstId = group.sourceIds[0];
  const lastId = group.sourceIds[group.sourceIds.length - 1];
  nodes.groupBadge.textContent = firstId === lastId
    ? `G ${firstId}${group.forcedBoundary ? " · forced" : ""}`
    : `G ${firstId}–${lastId}${group.forcedBoundary ? " · forced" : ""}`;
  nodes.groupBadge.classList.remove("d-none");
}

function pendingCharacterCount() {
  return pendingSegments.reduce((total, segment) => total + segment.sourceText.length, 0);
}

function pendingSegmentsLikelyReady() {
  if (pendingSegments.length === 0) return false;
  const first = pendingSegments[0];
  const last = pendingSegments[pendingSegments.length - 1];
  const hasPauseBoundary = pendingSegments.some((segment, index) => {
    if (index === 0) return false;
    const previous = pendingSegments[index - 1];
    const pause = segment.start - previous.end;
    return pause >= SOURCE_LONG_PAUSE_SECONDS
      || (
        pause >= SOURCE_WEAK_PUNCTUATION_PAUSE_SECONDS
        && SOURCE_WEAK_BOUNDARY_HINT_RE.test(previous.sourceText)
      );
  });
  return pendingSegments.some(segment => SOURCE_BOUNDARY_HINT_RE.test(segment.sourceText))
    || hasPauseBoundary
    || pendingSegments.length >= BATCH_SEGMENT_TRIGGER
    || pendingCharacterCount() >= BATCH_CHARACTER_TRIGGER
    || last.end - first.start >= SOURCE_GROUP_DURATION_TRIGGER_SECONDS;
}

function scheduleBatch() {
  if (batchTimer || pendingSegments.length === 0 || groupingRunning) return;
  batchTimer = window.setTimeout(() => {
    batchTimer = null;
    if (pendingSegmentsLikelyReady()) void flushPendingSegments();
  }, BATCH_DELAY_MS);
}

function takeGroupingSnapshot() {
  if (pendingSegments.length === 0) return [];
  const language = pendingSegments[0].language;
  const snapshot = [];
  let characters = 0;
  for (const candidate of pendingSegments) {
    if (snapshot.length >= BATCH_MAX_SEGMENTS) break;
    if (candidate.language !== language) break;
    const nextCharacters = characters + candidate.sourceText.length;
    if (snapshot.length > 0 && nextCharacters > GROUPING_MAX_CHARACTERS) break;
    snapshot.push(candidate);
    characters = nextCharacters;
  }
  return snapshot;
}

function groupingPayload(snapshot, final) {
  const groupNumber = ++groupingCounter;
  return {
    request_id: `youtube-${currentJobId}-group-${groupNumber}-${GROUPING_VERSION}`,
    source_language: snapshot[0].language,
    prompt_version: GROUPING_VERSION,
    final,
    segments: snapshot.map(segment => ({
      id: segment.id,
      text: segment.sourceText,
      start_ms: Math.round(segment.start * 1000),
      end_ms: Math.round(segment.end * 1000),
    })),
  };
}

function validateGroupingResponse(payload, data, snapshot) {
  if (
    !data
    || typeof data !== "object"
    || data.request_id !== payload.request_id
    || data.grouping_version !== GROUPING_VERSION
    || !Array.isArray(data.groups)
    || !Array.isArray(data.pending_tail_ids)
  ) {
    throw invalidTranslationResponseError();
  }

  const expectedIds = snapshot.map(segment => segment.id);
  const completeIds = [];
  const seenGroupIds = new Set();
  const segmentById = new Map(snapshot.map(segment => [segment.id, segment]));
  const groups = data.groups.map(group => {
    const sourceIds = Array.isArray(group?.source_ids)
      ? group.source_ids.map(Number)
      : [];
    if (
      typeof group?.group_id !== "string"
      || !group.group_id
      || seenGroupIds.has(group.group_id)
      || sourceIds.length === 0
      || sourceIds.some(id => !Number.isInteger(id) || !segmentById.has(id))
      || typeof group.source_text !== "string"
      || !group.source_text.trim()
    ) {
      throw invalidTranslationResponseError();
    }
    seenGroupIds.add(group.group_id);
    completeIds.push(...sourceIds);
    const segments = sourceIds.map(id => segmentById.get(id));
    if (
      contentSignature(group.source_text)
      !== contentSignature(segments.map(segment => segment.sourceText).join(""))
    ) {
      throw invalidTranslationResponseError();
    }
    return {
      groupId: group.group_id,
      sourceIds,
      sourceText: group.source_text.trim(),
      language: snapshot[0].language,
      forcedBoundary: group.forced_boundary === true,
      segments,
    };
  });
  const pendingTailIds = data.pending_tail_ids.map(Number);
  const returnedIds = [...completeIds, ...pendingTailIds];
  if (
    returnedIds.length !== expectedIds.length
    || returnedIds.some((id, index) => id !== expectedIds[index])
  ) {
    throw invalidTranslationResponseError();
  }
  return { groups, completedCount: completeIds.length };
}

function enqueueTranslationGroups(groups, { deferStart = false } = {}) {
  let batch = [];
  let characters = 0;
  let sourceIdCount = 0;
  const flushBatch = () => {
    if (batch.length === 0) return;
    enqueueTranslation(batch, { deferStart });
    batch = [];
    characters = 0;
    sourceIdCount = 0;
  };

  for (const rawGroup of groups) {
    const group = withPrecedingSourceContext(rawGroup, lastSourceGroupContext);
    lastSourceGroupContext = sourceGroupContext(group);
    for (const sourceId of group.sourceIds) {
      sourceTranslationGroups.set(sourceId, group);
      showSegmentTranslationGroup(sourceId, group);
    }
    const nextCharacters = characters + group.sourceText.length;
    const nextSourceIdCount = sourceIdCount + group.sourceIds.length;
    if (
      batch.length > 0
      && (
        nextCharacters > BATCH_MAX_CHARACTERS
        || nextSourceIdCount > BATCH_MAX_SEGMENTS
      )
    ) {
      flushBatch();
    }
    batch.push(group);
    characters += group.sourceText.length;
    sourceIdCount += group.sourceIds.length;
  }
  flushBatch();
}

function markGroupingFailure(snapshot, error) {
  const failureCode = translationErrorCode(error);
  for (const segment of snapshot) {
    failedSegmentIds.add(segment.id);
    translationFailureCodes.set(segment.id, failureCode);
    showSegmentTranslation(
      segment.id,
      "failed",
      error?.message || t("translationFailed"),
    );
  }
}

async function requestWorkflowWithRetries(operation, payload) {
  let lastError = null;
  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt += 1) {
    if (jobCancellationRequested) return null;
    try {
      return await requestTranslationWorkflow(operation, payload);
    } catch (error) {
      lastError = error;
      if (jobCancellationRequested) return null;
      if (!error.retryable || attempt >= RETRY_DELAYS_MS.length) break;
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAYS_MS[attempt]));
    }
  }
  throw lastError;
}

async function processPendingGroups() {
  while (pendingSegments.length > 0 && !jobCancellationRequested) {
    const final = finalGroupingRequested;
    const revision = pendingRevision;
    if (!final && revision === lastGroupedRevision) break;
    const snapshot = takeGroupingSnapshot();
    if (snapshot.length === 0) break;
    const hadMoreSegments = pendingSegments.length > snapshot.length;
    const payload = groupingPayload(snapshot, final);
    let data;
    try {
      data = await requestWorkflowWithRetries("group", payload);
      if (!data || jobCancellationRequested) return;
    } catch (error) {
      lastGroupedRevision = revision;
      if (!finalGroupingRequested) break;
      markGroupingFailure(snapshot, error);
      pendingSegments.splice(0, snapshot.length);
      pendingRevision += 1;
      continue;
    }

    const revisionChangedDuringRequest = pendingRevision !== revision;
    let result;
    try {
      result = validateGroupingResponse(payload, data, snapshot);
    } catch (error) {
      lastGroupedRevision = revision;
      if (!finalGroupingRequested) break;
      markGroupingFailure(snapshot, error);
      pendingSegments.splice(0, snapshot.length);
      pendingRevision += 1;
      continue;
    }
    if (result.completedCount > 0) {
      pendingSegments.splice(0, result.completedCount);
      pendingRevision += 1;
      enqueueTranslationGroups(result.groups, {
        deferStart: finalGroupingRequested,
      });
    }

    if (pendingSegments.length === 0) break;
    if (finalGroupingRequested && !final) continue;
    if (
      finalGroupingRequested
      || revisionChangedDuringRequest
      || (hadMoreSegments && result.completedCount > 0)
    ) continue;
    lastGroupedRevision = pendingRevision;
    break;
  }
}

function flushPendingSegments({ final = false } = {}) {
  if (batchTimer) {
    clearTimeout(batchTimer);
    batchTimer = null;
  }
  if (final) finalGroupingRequested = true;
  if (groupingRunning) return groupingPromise;
  if (pendingSegments.length === 0) {
    if (finalGroupingRequested) {
      finalGroupingRequested = false;
      startNextTranslationWave({ allowPartial: true });
    }
    updateTranslationProgress();
    return groupingPromise;
  }

  groupingRunning = true;
  groupingPromise = processPendingGroups()
    .catch(error => {
      if (!jobCancellationRequested) console.error("Source grouping failed", error);
    })
    .finally(() => {
      groupingRunning = false;
      if (finalGroupingRequested && pendingSegments.length === 0) {
        finalGroupingRequested = false;
        startNextTranslationWave({ allowPartial: true });
      } else if (
        !finalGroupingRequested
        && pendingSegments.length > 0
        && pendingRevision !== lastGroupedRevision
      ) {
        scheduleBatch();
      }
      updateTranslationProgress();
      maybeFinalize();
    });
  return groupingPromise;
}

function addPendingSegment(segment) {
  pendingSegments.push(segment);
  pendingRevision += 1;
  setTranslationActive(true);
  updateTranslationProgress();
  if (
    pendingSegments.length >= BATCH_SEGMENT_TRIGGER
    || pendingCharacterCount() >= BATCH_CHARACTER_TRIGGER
  ) {
    void flushPendingSegments();
  } else {
    scheduleBatch();
  }
}

function translationErrorMessage(data, fallback) {
  if (!data || typeof data !== "object") return fallback;
  if (data.error && typeof data.error.message === "string") return data.error.message;
  if (typeof data.detail === "string") return data.detail;
  if (data.detail && typeof data.detail.message === "string") return data.detail.message;
  if (typeof data.message === "string") return data.message;
  return fallback;
}

function responseIsRetryable(response, data, upstreamCode) {
  if (upstreamCode && NON_RETRYABLE_OUTPUT_CODES.has(String(upstreamCode))) return false;
  return RETRYABLE_HTTP_STATUSES.has(response.status)
    || data?.error?.retryable === true
    || data?.retryable === true
    || data?.detail?.retryable === true;
}

async function requestTranslationWorkflow(operation, payload) {
  let response;
  const controller = new AbortController();
  translationAbortControllers.add(controller);
  try {
    response = await fetch(`/api/youtube-live/translation-workflow/${operation}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Translation-Token": currentTranslationToken,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (error) {
    if (jobCancellationRequested || error.name === "AbortError") throw error;
    const wrapped = new Error(error.message || t("translationServiceFailed"));
    wrapped.retryable = true;
    wrapped.code = "NETWORK_ERROR";
    throw wrapped;
  } finally {
    translationAbortControllers.delete(controller);
  }

  let data = null;
  try {
    data = await response.json();
  } catch (_) {}

  if (!response.ok) {
    const error = new Error(translationErrorMessage(data, t("translationServiceFailed")));
    error.status = response.status;
    const upstreamCode = data?.code
      ?? data?.error_code
      ?? data?.error?.code
      ?? data?.detail?.code
      ?? data?.detail?.error_code;
    if (upstreamCode !== undefined && upstreamCode !== null) {
      error.code = String(upstreamCode);
    }
    error.retryable = responseIsRetryable(response, data, upstreamCode);
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
  for (let index = 0; index < batch.length; index += 1) {
    const group = batch[index];
    const result = data.translations[index];
    const sourceIds = Array.isArray(result?.source_ids)
      ? result.source_ids.map(Number)
      : [];
    if (
      result?.group_id !== group.groupId
      || sourceIds.length !== group.sourceIds.length
      || sourceIds.some((id, sourceIndex) => id !== group.sourceIds[sourceIndex])
      || typeof result.translated_text !== "string"
      || !result.translated_text.trim()
    ) {
      throw invalidTranslationResponseError();
    }
  }
  return data.translations;
}

async function translateBatch(batch) {
  const batchNumber = ++batchCounter;
  const requestId = `youtube-${currentJobId}-${selectedTargetLanguage}-group-batch-${batchNumber}-${GROUP_TRANSLATION_PROMPT_VERSION}`;
  const payload = {
    request_id: requestId,
    source_language: batch[0].language,
    target_language: selectedTargetLanguage,
    prompt_version: GROUP_TRANSLATION_PROMPT_VERSION,
    on_screen_terms: extractTitleTerms(videoTitle.textContent),
    preceding_source_context: batch[0].precedingSourceContext || [],
    groups: batch.map(group => ({
      group_id: group.groupId,
      source_ids: group.sourceIds,
      source_text: group.sourceText,
      low_confidence_spans: [...new Set(
        group.segments.flatMap(segment => segment.lowConfidenceSpans || []),
      )],
      segments: group.segments.map(segment => ({
        id: segment.id,
        source_text: segment.sourceText,
        start_ms: Math.round(segment.start * 1000),
        end_ms: Math.round(segment.end * 1000),
        words: (segment.words || [])
          .map(word => ({
            word: word.word,
            start_ms: Math.max(
              Math.round(segment.start * 1000),
              Math.round(word.start * 1000),
            ),
            end_ms: Math.min(
              Math.round(segment.end * 1000),
              Math.round(word.end * 1000),
            ),
          }))
          .filter(word => word.word.trim() && word.end_ms > word.start_ms),
      })),
    })),
  };

  let lastError = null;
  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt += 1) {
    if (jobCancellationRequested) return;
    try {
      const data = await requestTranslationWorkflow("translate-groups", payload);
      const results = validateTranslationResponse(payload, data, batch);
      for (let index = 0; index < results.length; index += 1) {
        const result = results[index];
        const group = batch[index];
        const translatedText = result.translated_text.trim();
        const displayCues = validateDisplayCues(group, result);
        translatedGroups.set(group.groupId, {
          groupId: group.groupId,
          sourceIds: [...group.sourceIds],
          translatedText,
          forcedBoundary: group.forcedBoundary,
        });
        for (let cueIndex = translatedDisplayCues.length - 1; cueIndex >= 0; cueIndex -= 1) {
          if (translatedDisplayCues[cueIndex].groupId === group.groupId) {
            translatedDisplayCues.splice(cueIndex, 1);
          }
        }
        translatedDisplayCues.push(...displayCues);
        translatedDisplayCues.sort((left, right) => (
          left.start - right.start || left.end - right.end || left.cueId.localeCompare(right.cueId)
        ));
        for (const sourceId of group.sourceIds) {
          const displayText = translationForSourceId(displayCues, sourceId) || translatedText;
          translatedSegments.set(sourceId, { id: sourceId, translatedText: displayText });
          failedSegmentIds.delete(sourceId);
          translationFailureCodes.delete(sourceId);
          showSegmentTranslation(sourceId, "ready", displayText);
        }
      }
      return;
    } catch (error) {
      lastError = error;
      if (jobCancellationRequested) return;
      if (!error.retryable || attempt >= RETRY_DELAYS_MS.length) break;
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAYS_MS[attempt]));
    }
  }

  if (
    batch.length > 1
    && NON_RETRYABLE_OUTPUT_CODES.has(String(lastError?.code || ""))
  ) {
    const midpoint = Math.ceil(batch.length / 2);
    await translateBatch(batch.slice(0, midpoint));
    await translateBatch(batch.slice(midpoint));
    return;
  }

  const failureCode = translationErrorCode(lastError);
  for (const group of batch) {
    for (const segment of group.segments) {
      failedSegmentIds.add(segment.id);
      translationFailureCodes.set(segment.id, failureCode);
      showSegmentTranslation(
        segment.id,
        "failed",
        lastError?.message || t("translationFailed"),
      );
    }
  }
}

async function processTranslationBatch(batch) {
  if (jobCancellationRequested) return;
  try {
    await translateBatch(batch);
  } catch (error) {
    if (jobCancellationRequested) return;
    console.error("Translation queue failed", error);
    const failureCode = translationErrorCode(error);
    for (const group of batch) {
      for (const segment of group.segments) {
        failedSegmentIds.add(segment.id);
        translationFailureCodes.set(segment.id, failureCode);
        showSegmentTranslation(segment.id, "failed", error.message || t("translationFailed"));
      }
    }
  } finally {
    queuedBatchCount = Math.max(0, queuedBatchCount - 1);
    updateTranslationProgress();
  }
}

function startNextTranslationWave({ allowPartial = false } = {}) {
  if (translationWaveRunning || jobCancellationRequested) return translationQueue;
  if (pendingTranslationBatches.length === 0) return translationQueue;
  if (!allowPartial && pendingTranslationBatches.length < TRANSLATION_WAVE_SIZE) {
    return translationQueue;
  }

  const wave = pendingTranslationBatches.splice(0, TRANSLATION_WAVE_SIZE);
  translationWaveRunning = true;
  translationQueue = Promise.allSettled(wave.map(processTranslationBatch)).finally(() => {
    translationWaveRunning = false;
    if (!jobCancellationRequested) {
      startNextTranslationWave({ allowPartial: transcriptionDone });
      maybeFinalize();
    }
  });
  return translationQueue;
}

function enqueueTranslation(batch, { deferStart = false } = {}) {
  if (!batch.length || jobCancellationRequested) return translationQueue;
  if (translationLanguagesMatch(batch[0]?.language, selectedTargetLanguage)) {
    for (const group of batch) {
      for (const segment of group.segments) {
        translatedSegments.set(segment.id, {
          id: segment.id,
          translatedText: segment.sourceText,
        });
        showSegmentTranslation(segment.id, "ready", segment.sourceText);
      }
    }
    if (!sameLanguageNoticeShown) {
      sameLanguageNoticeShown = true;
      setStatus(t("sameLanguage"), "running");
    }
    updateTranslationProgress();
    return translationQueue;
  }
  queuedBatchCount += 1;
  pendingTranslationBatches.push(batch);
  setTranslationActive(true);
  for (const group of batch) {
    for (const segment of group.segments) showSegmentTranslation(segment.id, "pending");
  }
  if (!deferStart) startNextTranslationWave({ allowPartial: transcriptionDone });
  return translationQueue;
}

function abortTranslationRequests() {
  for (const controller of translationAbortControllers) controller.abort();
  translationAbortControllers.clear();
}

function effectiveSourceLanguage(dataLanguage) {
  return requestedSourceLanguage || normalizeTranslationLanguage(dataLanguage);
}

function addSegment(data) {
  if (jobCancellationRequested) return;
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
    words: Array.isArray(data.words)
      ? data.words
        .map(word => ({
          word: String(word?.word || ""),
          start: Number(word?.start),
          end: Number(word?.end),
          probability: Number(word?.probability),
        }))
        .filter(word => word.word.trim() && Number.isFinite(word.start) && Number.isFinite(word.end))
      : [],
  };
  sourceSegments.set(id, segment);
  playbackSegments.push(segment);
  if (
    playbackSegments.length > 1
    && playbackSegments[playbackSegments.length - 2].start > segment.start
  ) {
    playbackSegments.sort((left, right) => left.start - right.start);
  }
  renderSegment(segment);
  updateTranscriptionProgress(data);

  if (!normalizedLanguage) {
    const rawLanguage = segment.rawLanguage || "unknown";
    unsupportedLanguageNotice = t("unsupportedLanguage", { language: rawLanguage });
    failedSegmentIds.add(id);
    translationFailureCodes.set(id, "UNSUPPORTED_LANGUAGE");
    showSegmentTranslation(id, "failed", unsupportedLanguageNotice);
    setStatus(appendTranslationErrorCodes(unsupportedLanguageNotice), "failed");
  } else if (translationLanguagesMatch(normalizedLanguage, selectedTargetLanguage)) {
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
  if (jobCancellationRequested) {
    setTranslationActive(false);
    setTranscriptionActive(false);
    return;
  }
  if (!transcriptionDone || pendingSegments.length > 0 || queuedBatchCount > 0) {
    setTranslationActive(pendingSegments.length > 0 || queuedBatchCount > 0);
    return;
  }
  setTranslationActive(false);
  setTranscriptionActive(false);
  currentCancelRequest = null;
  cancelJob.classList.add("d-none");
  if (uploadMode && uploadBatchJobs.length && uploadBatchJobIndex >= 0) {
    const batchJob = uploadBatchJobs[uploadBatchJobIndex];
    const fileIndex = Number(batchJob.batch_index);
    uploadBatchCancelRequests.delete(batchJob.job_id);
    if (transcriptionFailed) {
      setUploadFileStatus(fileIndex, "videoBatchFailed", "failed");
    } else {
      addUploadBatchDownload(fileIndex, batchJob.filename);
      setUploadFileStatus(fileIndex, "videoBatchDone", "done");
    }
    const nextIndex = uploadBatchJobIndex + 1;
    if (nextIndex < uploadBatchJobs.length && !uploadBatchAdvanceScheduled) {
      uploadBatchAdvanceScheduled = true;
      setTimeout(() => {
        uploadBatchAdvanceScheduled = false;
        activateUploadBatchJob(nextIndex);
      }, 0);
      return;
    }
  }
  updateTranslationProgress();
  updateActionButtons();
  actionBar.classList.toggle(
    "d-none",
    uploadMode && uploadBatchJobs.length > 1,
  );
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
  if (mode === "source") {
    const entries = normalizeSrtTimeline(sortedSourceSegments().map(segment => ({
      start: segment.start,
      end: segment.end,
      text: segment.sourceText,
    })));
    return entries.map((segment, index) => (
      `${index + 1}\n${formatSrtTimestamp(segment.start)} --> ${formatSrtTimestamp(segment.end)}\n${segment.text}`
    )).join("\n\n") + (entries.length ? "\n" : "");
  }

  const cueSourceIds = new Set(
    translatedDisplayCues.flatMap(cue => cue.sourceIds),
  );
  const entries = sortedDisplayCues().map(cue => {
    const translation = cue.lines.join("\n");
    const sourceText = sourceTextForDisplayCue(cue);
    return {
      start: cue.start,
      end: cue.end,
      text: mode === "bilingual" && sourceText
        ? `${translation}\n${sourceText}`
        : translation,
    };
  });
  for (const segment of sortedSourceSegments()) {
    if (cueSourceIds.has(segment.id)) continue;
    const translation = translatedSegments.get(segment.id)?.translatedText
      || segment.sourceText;
    entries.push({
      start: segment.start,
      end: segment.end,
      text: mode === "bilingual"
        ? `${translation}\n${segment.sourceText}`
        : translation,
    });
  }
  const normalizedEntries = normalizeSrtTimeline(entries);
  return normalizedEntries.map((entry, index) => (
    `${index + 1}\n${formatSrtTimestamp(entry.start)} --> ${formatSrtTimestamp(entry.end)}\n${entry.text}`
  )).join("\n\n") + (normalizedEntries.length ? "\n" : "");
}

function buildSegmentsJson() {
  return JSON.stringify({
    schema_version: 4,
    source_language: requestedSourceLanguage || detectedSourceLanguage || null,
    target_language: selectedTargetLanguage,
    transcription_mode: transcriptionMode.value,
    translation_groups: [...translatedGroups.values()].map(group => ({
      group_id: group.groupId,
      source_ids: group.sourceIds,
      translated_text: group.translatedText,
      forced_boundary: group.forcedBoundary,
    })),
    display_cues: sortedDisplayCues().map(cue => ({
      cue_id: cue.cueId,
      group_id: cue.groupId,
      source_ids: cue.sourceIds,
      start_ms: Math.round(cue.start * 1000),
      end_ms: Math.round(cue.end * 1000),
      source_text: cue.sourceText,
      source_lines: cue.sourceLines,
      translated_text: cue.translatedText,
      lines: cue.lines,
    })),
    segments: sortedSourceSegments().map(segment => {
      const group = sourceTranslationGroups.get(segment.id);
      return {
        id: segment.id,
        start_ms: Math.round(segment.start * 1000),
        end_ms: Math.round(segment.end * 1000),
        source_text: segment.sourceText,
        translated_text: translatedSegments.get(segment.id)?.translatedText
          || segment.sourceText,
        ...(
          group
            ? {
              translation_group_id: group.groupId,
              translation_group_source_ids: group.sourceIds,
              translation_group_forced_boundary: group.forcedBoundary,
            }
            : {}
        ),
        words: segment.words.map(word => ({
          word: word.word,
          start_ms: Math.round(word.start * 1000),
          end_ms: Math.round(word.end * 1000),
          ...(
            Number.isFinite(word.probability)
              ? { probability: word.probability }
              : {}
          ),
        })),
      };
    }),
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

function addUploadBatchDownload(index, filename) {
  if (uploadBatchResults.has(index) || transcriptionFailed) return;
  const safeBase = filename.replace(/\.[^.]+$/, "").replace(/[\\/:*?"<>|]+/g, "_") || "video";
  const result = {
    filename,
    source: buildSrt("source"),
    translated: buildSrt("translated"),
    bilingual: buildSrt("bilingual"),
    json: buildSegmentsJson(),
  };
  uploadBatchResults.set(index, result);

  const item = document.createElement("div");
  item.className = "upload-download-item";
  const name = document.createElement("div");
  name.className = "upload-download-name";
  name.textContent = filename;
  name.title = filename;
  const actions = document.createElement("div");
  actions.className = "d-flex flex-wrap gap-2";
  [
    ["downloadSourceSrt", result.source, `${safeBase}_source.srt`, "application/x-subrip;charset=utf-8"],
    ["downloadTranslatedSrt", result.translated, `${safeBase}_translated.srt`, "application/x-subrip;charset=utf-8"],
    ["downloadBilingualSrt", result.bilingual, `${safeBase}_bilingual.srt`, "application/x-subrip;charset=utf-8"],
    ["downloadJson", result.json, `${safeBase}_segments.json`, "application/json;charset=utf-8"],
  ].forEach(([labelKey, content, downloadName, type]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = labelKey === "downloadBilingualSrt"
      ? "btn btn-sm btn-primary"
      : "btn btn-sm btn-outline-secondary";
    button.dataset.i18n = labelKey;
    button.textContent = t(labelKey);
    button.addEventListener("click", () => downloadTextFile(content, downloadName, type));
    actions.append(button);
  });
  item.append(name, actions);
  uploadDownloadList.append(item);
  uploadBatchDownloads.classList.remove("d-none");
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
      focusPlaybackSegment(playbackSegmentAt(seekTime), seekTime);
    }
    return;
  }

  const button = event.target.closest("[data-retry-segment]");
  if (!button) return;
  const segmentId = Number(button.dataset.retrySegment);
  const segment = sourceSegments.get(segmentId);
  if (!segment || !segment.language) return;
  const group = sourceTranslationGroups.get(segmentId) || {
    groupId: `retry-${segment.id}`,
    sourceIds: [segment.id],
    sourceText: segment.sourceText,
    language: segment.language,
    forcedBoundary: true,
    segments: [segment],
    precedingSourceContext: [],
  };
  for (const sourceId of group.sourceIds) {
    failedSegmentIds.delete(sourceId);
    translationFailureCodes.delete(sourceId);
    translatedSegments.delete(sourceId);
    showSegmentTranslation(sourceId, "pending");
  }
  setTranslationActive(true);
  actionBar.classList.add("d-none");
  enqueueTranslation([group]);
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
    setStatus(t(uploadMode ? "waitingUpload" : "waiting"), "idle");
  } catch (error) {
    setStatus(error.message || t("captchaLoadFailed"), "failed");
  }
});
verifyCaptcha.addEventListener("click", verifyCaptchaAnswer);
cancelJob.addEventListener("click", async () => {
  if (uploadMode && uploadBatchCancelRequests.size) {
    cancelJob.disabled = true;
    const requests = [...uploadBatchCancelRequests.entries()];
    await Promise.allSettled(requests.map(async ([jobId, cancelRequest]) => {
      const response = await fetch(cancelRequest.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cancel_token: cancelRequest.token }),
      });
      if (!response.ok) throw new Error(await readError(response));
      const job = uploadBatchJobs.find(item => item.job_id === jobId);
      if (job) setUploadFileStatus(Number(job.batch_index), "videoBatchCancelled", "failed");
    }));
    uploadBatchCancelRequests.clear();
    jobCancellationRequested = true;
    currentCancelRequest = null;
    currentTranslationToken = "";
    pendingSegments = [];
    pendingTranslationBatches = [];
    queuedBatchCount = 0;
    if (batchTimer) clearTimeout(batchTimer);
    batchTimer = null;
    abortTranslationRequests();
    if (eventSource) eventSource.close();
    eventSource = null;
    transcriptionDone = true;
    setTranscriptionActive(false);
    setTranslationActive(false);
    cancelJob.classList.add("d-none");
    transcriptionBar.classList.remove("progress-bar-animated");
    translationBar.classList.remove("progress-bar-animated");
    languageConfirmation.classList.add("d-none");
    sourceLanguage.disabled = false;
    targetLanguage.disabled = false;
    setSourceControlsDisabled(false);
    setStatus(t("cancelled"), "idle");
    resetCaptchaState(false);
    notifyParentHeight();
    return;
  }
  if (!currentCancelRequest) return;
  const cancelRequest = currentCancelRequest;
  cancelJob.disabled = true;
  try {
    const response = await fetch(cancelRequest.url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cancel_token: cancelRequest.token }),
    });
    if (!response.ok) throw new Error(await readError(response));

    jobCancellationRequested = true;
    currentCancelRequest = null;
    currentTranslationToken = "";
    pendingSegments = [];
    pendingTranslationBatches = [];
    queuedBatchCount = 0;
    if (batchTimer) clearTimeout(batchTimer);
    batchTimer = null;
    abortTranslationRequests();
    if (eventSource) eventSource.close();
    eventSource = null;
    transcriptionDone = true;
    setTranscriptionActive(false);
    setTranslationActive(false);
    cancelJob.classList.add("d-none");
    transcriptionBar.classList.remove("progress-bar-animated");
    translationBar.classList.remove("progress-bar-animated");
    languageConfirmation.classList.add("d-none");
    sourceLanguage.disabled = false;
    targetLanguage.disabled = false;
    setSourceControlsDisabled(false);
    setStatus(t("cancelled"), "idle");
    resetCaptchaState(false);
    notifyParentHeight();
  } catch (error) {
    cancelJob.disabled = false;
    setStatus(error.message || t("requestFailed"), "failed");
  }
});

function connectTranscriptionJob(job, uploadFilename = "") {
  if (!job.job_id || !job.events_url || !job.translation_token
    || !job.cancel_url || !job.cancel_token) {
    throw new Error(t("requestFailed"));
  }
  currentJobId = job.job_id;
  currentTranslationToken = job.translation_token;
  currentCancelRequest = {
    url: job.cancel_url,
    token: job.cancel_token,
  };
  cancelJob.classList.remove("d-none");

  eventSource = new EventSource(job.events_url);
  eventSource.addEventListener("status", event => {
    const data = JSON.parse(event.data);
    setStatus(uploadMode ? t("processing") : (data.message || t("processing")), "running");
  });
  eventSource.addEventListener("metadata", event => {
    const data = JSON.parse(event.data);
    const playerDuration = Number(youtubePlayerController?.getDuration?.()) || 0;
    totalDuration = Number(data.duration) || playerDuration;
    videoTitle.textContent = data.title
      || uploadFilename
      || t(uploadMode ? "uploadedVideoTitle" : "videoTitle");
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
  eventSource.addEventListener("progress", event => {
    try {
      updateTranscriptionProgress(JSON.parse(event.data || "{}"));
    } catch (error) {
      console.error("Could not process transcription progress", error);
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
    setTranscriptionProgress(100, t("transcriptionDone"));
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
  eventSource.addEventListener("cancelled", () => {
    transcriptionDone = true;
    transcriptionFailed = false;
    currentCancelRequest = null;
    currentTranslationToken = "";
    jobCancellationRequested = true;
    pendingSegments = [];
    pendingTranslationBatches = [];
    queuedBatchCount = 0;
    if (batchTimer) clearTimeout(batchTimer);
    batchTimer = null;
    cancelJob.classList.add("d-none");
    abortTranslationRequests();
    setTranscriptionActive(false);
    setTranslationActive(false);
    transcriptionBar.classList.remove("progress-bar-animated");
    translationBar.classList.remove("progress-bar-animated");
    setStatus(t("cancelled"), "idle");
    if (eventSource) eventSource.close();
    eventSource = null;
    maybeFinalize();
  });
  eventSource.onerror = () => {
    if (!eventSource) return;
    cancelCurrentJob();
    jobCancellationRequested = !(uploadMode && uploadBatchJobs.length);
    pendingSegments = [];
    pendingTranslationBatches = [];
    queuedBatchCount = 0;
    if (batchTimer) clearTimeout(batchTimer);
    batchTimer = null;
    cancelJob.classList.add("d-none");
    abortTranslationRequests();
    transcriptionDone = true;
    transcriptionFailed = true;
    setTranscriptionActive(false);
    setStatus(t("disconnected"), "failed");
    eventSource.close();
    eventSource = null;
    flushPendingSegments({ final: true });
    maybeFinalize();
  };
}

function activateUploadBatchJob(position) {
  const job = uploadBatchJobs[position];
  if (!job) return;
  if (position > 0) resetView();
  uploadBatchJobIndex = position;
  selectedUploadFileIndex = Number(job.batch_index);
  selectUploadFile(selectedUploadFileIndex);
  setUploadFileStatus(selectedUploadFileIndex, "videoBatchProcessing", "running");
  setTranscriptionActive(true);
  setStatus(t("processing"), "running");
  if (requestedSourceLanguage) revealResults();
  try {
    connectTranscriptionJob(job, job.filename);
  } catch (error) {
    transcriptionDone = true;
    transcriptionFailed = true;
    setTranscriptionActive(false);
    setStatus(error.message || t("requestFailed"), "failed");
    maybeFinalize();
  }
}

async function beginTranscription() {
  requestedSourceLanguage = sourceLanguage.value;
  selectedTargetLanguage = targetLanguage.value;
  if (uploadMode && videoFile.files.length > MAX_UPLOAD_VIDEO_FILES) {
    setStatus(t("videoFileLimit"), "failed");
    resetCaptchaState(false);
    return;
  }
  resetView();
  if (uploadMode) {
    uploadBatchJobs = [];
    uploadBatchJobIndex = -1;
    uploadBatchAdvanceScheduled = false;
    uploadBatchCancelRequests.clear();
    uploadBatchResults.clear();
    uploadDownloadList.replaceChildren();
    uploadBatchDownloads.classList.add("d-none");
    uploadFileStates.clear();
    renderUploadFileSelector();
  }
  sourceLanguage.disabled = true;
  targetLanguage.disabled = true;
  setSourceControlsDisabled(true);
  startBtn.disabled = true;
  startBtn.classList.add("d-none");
  captchaBlock.classList.add("d-none");
  setTranscriptionActive(true);
  setStatus(t(uploadMode ? "uploading" : "creating"), "running");
  if (requestedSourceLanguage) revealResults();

  const whisperLanguage = requestedSourceLanguage === "zh-TW"
    ? "zh"
    : requestedSourceLanguage;

  try {
    let response;
    if (uploadMode) {
      const uploadBody = new FormData();
      [...videoFile.files].forEach(file => uploadBody.append("files", file));
      uploadBody.append("language", whisperLanguage);
      uploadBody.append("include_word_timestamps", String(includeWordTimestamps.checked));
      uploadBody.append("transcription_mode", transcriptionMode.value);
      uploadBody.append("captcha_token", captchaToken.value);
      response = await fetch("/api/video-upload/jobs/batch", {
        method: "POST",
        body: uploadBody,
      });
    } else {
      response = await fetch("/api/youtube-live/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: youtubeUrl.value.trim(),
          language: whisperLanguage,
          ignore_subtitles: ignoreSubtitles.checked,
          include_word_timestamps: includeWordTimestamps.checked,
          transcription_mode: transcriptionMode.value,
          captcha_token: captchaToken.value,
        }),
      });
    }
    if (!response.ok) throw new Error(await readError(response));
    if (uploadMode) {
      const batch = await response.json();
      for (const failure of batch.errors || []) {
        setUploadFileStatus(Number(failure.batch_index), "videoBatchFailed", "failed");
      }
      uploadBatchJobs = [...(batch.jobs || [])].sort(
        (left, right) => Number(left.batch_index) - Number(right.batch_index),
      );
      for (const job of uploadBatchJobs) {
        setUploadFileStatus(Number(job.batch_index), "videoBatchQueued", "queued");
        uploadBatchCancelRequests.set(job.job_id, {
          url: job.cancel_url,
          token: job.cancel_token,
        });
      }
      if (!uploadBatchJobs.length) {
        throw new Error(batch.errors?.[0]?.error || t("requestFailed"));
      }
      activateUploadBatchJob(0);
      return;
    }
    const job = await response.json();
    if (!job.job_id || !job.events_url || !job.translation_token
      || !job.cancel_url || !job.cancel_token) {
      throw new Error(t("requestFailed"));
    }
    currentJobId = job.job_id;
    currentTranslationToken = job.translation_token;
    currentCancelRequest = {
      url: job.cancel_url,
      token: job.cancel_token,
    };
    cancelJob.classList.remove("d-none");

    eventSource = new EventSource(job.events_url);
    eventSource.addEventListener("status", event => {
      const data = JSON.parse(event.data);
      setStatus(uploadMode ? t("processing") : (data.message || t("processing")), "running");
    });
    eventSource.addEventListener("metadata", event => {
      const data = JSON.parse(event.data);
      const playerDuration = Number(youtubePlayerController?.getDuration?.()) || 0;
      totalDuration = Number(data.duration) || playerDuration;
      videoTitle.textContent = data.title
        || (uploadMode ? videoFile.files?.[0]?.name : "")
        || t(uploadMode ? "uploadedVideoTitle" : "videoTitle");
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
    eventSource.addEventListener("progress", event => {
      try {
        updateTranscriptionProgress(JSON.parse(event.data || "{}"));
      } catch (error) {
        console.error("Could not process transcription progress", error);
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
      setTranscriptionProgress(100, t("transcriptionDone"));
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
    eventSource.addEventListener("cancelled", event => {
      transcriptionDone = true;
      transcriptionFailed = false;
      currentCancelRequest = null;
      currentTranslationToken = "";
      jobCancellationRequested = true;
      pendingSegments = [];
      pendingTranslationBatches = [];
      queuedBatchCount = 0;
      if (batchTimer) clearTimeout(batchTimer);
      batchTimer = null;
      cancelJob.classList.add("d-none");
      abortTranslationRequests();
      setTranscriptionActive(false);
      setTranslationActive(false);
      transcriptionBar.classList.remove("progress-bar-animated");
      translationBar.classList.remove("progress-bar-animated");
      setStatus(t("cancelled"), "idle");
      if (eventSource) eventSource.close();
      eventSource = null;
      maybeFinalize();
    });
    eventSource.onerror = () => {
      if (!eventSource) return;
      cancelCurrentJob();
      jobCancellationRequested = true;
      pendingSegments = [];
      pendingTranslationBatches = [];
      queuedBatchCount = 0;
      if (batchTimer) clearTimeout(batchTimer);
      batchTimer = null;
      cancelJob.classList.add("d-none");
      abortTranslationRequests();
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
    if (currentCancelRequest) cancelCurrentJob();
    cancelJob.classList.add("d-none");
    transcriptionDone = true;
    transcriptionFailed = true;
    setTranscriptionActive(false);
    setStatus(error.message || t("requestFailed"), "failed");
    sourceLanguage.disabled = false;
    targetLanguage.disabled = false;
    setSourceControlsDisabled(false);
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
    setStatus(t(uploadMode ? "processing" : "creating"), "running");
  } catch (error) {
    setStatus(error.message || t("requestFailed"), "failed");
    confirmLanguageBtn.disabled = false;
  }
});

if (uploadMode) {
  let dragDepth = 0;
  videoFile.addEventListener("change", () => {
    setSelectedVideoFiles(videoFile.files);
  });
  videoDropZone.addEventListener("dragenter", event => {
    if (!dragIncludesFiles(event)) return;
    event.preventDefault();
    if (videoFile.disabled) return;
    dragDepth += 1;
    videoDropZone.classList.add("dragging");
  });
  videoDropZone.addEventListener("dragover", event => {
    if (!dragIncludesFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = videoFile.disabled ? "none" : "copy";
  });
  videoDropZone.addEventListener("dragleave", event => {
    if (!dragIncludesFiles(event)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) videoDropZone.classList.remove("dragging");
  });
  videoDropZone.addEventListener("drop", event => {
    if (!dragIncludesFiles(event)) return;
    event.preventDefault();
    dragDepth = 0;
    videoDropZone.classList.remove("dragging");
    if (videoFile.disabled) return;
    setSelectedVideoFiles(event.dataTransfer?.files || [], { append: true });
  });
} else {
  youtubeUrl.addEventListener("input", scheduleVideoPreview);
  youtubeUrl.addEventListener("change", updateVideoPreview);
}
sourceLanguage.addEventListener("change", updateStartButton);
videoFullscreenButton.addEventListener("click", toggleVideoFullscreen);
document.addEventListener("fullscreenchange", updateFullscreenButton);
document.addEventListener("webkitfullscreenchange", updateFullscreenButton);

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (uploadMode && !videoFile.files.length) {
    setStatus(t("waitingUpload"), "failed");
    videoFile.focus();
    return;
  }
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

configureSourceMode();
applyLanguage(currentLanguage);
resetView();
resetCaptchaState(false);
setStatus(t(uploadMode ? "waitingUpload" : "waiting"), "idle");
loadPublicConfig();
notifyParentHeight();
window.addEventListener("load", notifyParentHeight);
window.addEventListener("resize", notifyParentHeight);
