const API_URL = "/api/dashboard/status";
const TOKEN_KEY = "aimarumaru-dashboard-token";
const POLL_MS = 2000;

const $ = id => document.getElementById(id);
const state = { token: sessionStorage.getItem(TOKEN_KEY) || "", timer: null, gpuHistory: [] };

function number(value, digits = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toLocaleString("zh-TW", { maximumFractionDigits: digits }) : "—";
}

function duration(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  if (value < 60) return `${value.toFixed(value < 10 ? 1 : 0)} 秒`;
  const minutes = Math.floor(value / 60);
  const remainder = Math.floor(value % 60);
  return `${minutes} 分 ${remainder} 秒`;
}

function bytes(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "—";
  const mib = parsed > 1024 * 1024 ? parsed / 1024 / 1024 : parsed;
  return `${(mib / 1024).toFixed(1)} GB`;
}

function setConnection(live, message) {
  $("connectionDot").classList.toggle("live", live);
  $("connectionText").textContent = message;
}

function showLogin(message = "") {
  $("loginPanel").classList.remove("hidden");
  $("dashboard").classList.add("hidden");
  $("loginError").textContent = message;
  setConnection(false, "需要授權");
}

function showDashboard() {
  $("loginPanel").classList.add("hidden");
  $("dashboard").classList.remove("hidden");
  setConnection(true, "即時連線");
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderSparkline(value) {
  state.gpuHistory.push(Math.max(0, Math.min(100, Number(value) || 0)));
  state.gpuHistory = state.gpuHistory.slice(-60);
  const denominator = Math.max(1, state.gpuHistory.length - 1);
  const points = state.gpuHistory.map((item, index) => `${index / denominator * 100},${40 - item * .4}`).join(" ");
  $("gpuSpark").querySelector("polyline").setAttribute("points", points);
}

function renderGpu(gpus) {
  const list = Array.isArray(gpus) ? gpus : [];
  const averageLoad = list.length ? list.reduce((sum, gpu) => sum + (Number(gpu.utilization_gpu) || 0), 0) / list.length : 0;
  const used = list.reduce((sum, gpu) => sum + (Number(gpu.memory_used) || 0), 0);
  const total = list.reduce((sum, gpu) => sum + (Number(gpu.memory_total) || 0), 0);
  $("gpuValue").textContent = list.length ? `${averageLoad.toFixed(0)}%` : "N/A";
  $("gpuNote").textContent = list.length ? `${list.length} 張 GPU` : "NVML 無資料";
  $("vramValue").textContent = total ? `${bytes(used)} / ${bytes(total)}` : "N/A";
  $("vramNote").textContent = total ? `使用率 ${(used / total * 100).toFixed(0)}%` : "NVML 無資料";
  renderSparkline(averageLoad);

  const container = $("gpus");
  container.replaceChildren();
  if (!list.length) { container.append(element("div", "empty", "目前無法取得 NVIDIA GPU 資訊")); return; }
  for (const gpu of list) {
    const card = element("article", "gpu");
    const title = element("div", "gpu-title");
    title.append(element("span", "", `GPU ${gpu.index} · ${gpu.name || "NVIDIA"}`));
    title.append(element("span", "", `${number(gpu.temperature_gpu)}°C`));
    card.append(title);
    const bars = element("div", "gpu-bars");
    const load = Math.max(0, Math.min(100, Number(gpu.utilization_gpu) || 0));
    const memory = Number(gpu.memory_total) ? Number(gpu.memory_used) / Number(gpu.memory_total) * 100 : 0;
    for (const [label, value] of [["GPU", load], ["VRAM", memory]]) {
      const row = element("div", "bar-row");
      row.append(element("span", "", label));
      const bar = element("div", "bar");
      const fill = element("span"); fill.style.width = `${value}%`; bar.append(fill);
      row.append(bar, element("span", "", `${value.toFixed(0)}%`));
      bars.append(row);
    }
    card.append(bars);
    container.append(card);
  }
}

function stat(label, value) {
  const node = element("div", "job-stat");
  node.append(element("b", "", value), element("span", "", label));
  return node;
}

function renderJobs(jobs) {
  const list = Array.isArray(jobs) ? jobs : [];
  const container = $("jobs");
  container.replaceChildren();
  $("jobsMeta").textContent = `${list.length} 筆，僅顯示遙測資料`;
  if (!list.length) { container.append(element("div", "empty", "尚無轉錄工作")); return; }
  for (const job of list) {
    const transcribe = job.transcription || {};
    const translate = job.translation || {};
    const card = element("article", "job");
    const top = element("div", "job-top");
    top.append(element("div", "job-id", job.job_id));
    const badgeText = job.stalled ? "stalled" : (job.status || "unknown");
    top.append(element("span", `badge ${badgeText}`, badgeText));
    card.append(top);
    const meta = element("div", "job-meta");
    meta.append(
      element("span", "", `${job.processing_profile} / ${job.asr_provider}`),
      element("span", "", `階段 ${job.phase}`),
      element("span", "", `語言 ${job.source_language}`),
      element("span", "", `最後活動 ${duration(job.last_activity_age_seconds)}前`),
    );
    card.append(meta);
    const progress = Math.max(0, Math.min(100, Number(transcribe.progress_percent) || 0));
    const progressNode = element("div", "progress");
    const fill = element("span"); fill.style.width = `${progress}%`; progressNode.append(fill); card.append(progressNode);
    const stats = element("div", "job-stats");
    stats.append(
      stat("轉錄進度", `${progress.toFixed(1)}%`),
      stat("最後 Chunk", transcribe.last_chunk_ms == null ? "—" : duration(Number(transcribe.last_chunk_ms) / 1000)),
      stat("音訊等待", transcribe.last_input_wait_ms == null ? "—" : duration(Number(transcribe.last_input_wait_ms) / 1000)),
      stat("Whisper 推論", transcribe.last_inference_ms == null ? "—" : duration(Number(transcribe.last_inference_ms) / 1000)),
      stat("事件送出", transcribe.last_event_emit_ms == null ? "—" : duration(Number(transcribe.last_event_emit_ms) / 1000)),
      stat("處理倍率", transcribe.processing_speed_x == null ? "—" : `${number(transcribe.processing_speed_x, 2)}×`),
      stat("翻譯 P95", translate.p95_latency_ms == null ? "—" : duration(Number(translate.p95_latency_ms) / 1000)),
      stat("字幕段數", number(transcribe.segments_emitted)),
      stat("翻譯來源 ID", `${number(translate.source_ids_succeeded)} / ${number(translate.source_ids_seen)}`),
      stat("翻譯錯誤", number(translate.failed_requests)),
      stat("估算費用", `US$${number(translate.estimated_cost_usd, 5)}`),
    );
    card.append(stats);
    container.append(card);
  }
}

function render(data) {
  showDashboard();
  renderGpu(data.gpus);
  renderJobs(data.jobs);
  const queue = data.queue || {};
  const summary = data.summary || {};
  $("queueValue").textContent = `${number(queue.waiting_count)} / ${number(queue.transcribing_count)}`;
  $("queueNote").textContent = "等待 / 處理中";
  const healthy = Number(summary.stalled_jobs || 0) === 0;
  $("healthValue").textContent = healthy ? "Healthy" : `${number(summary.stalled_jobs)} Stalled`;
  $("healthValue").style.color = healthy ? "var(--green)" : "var(--red)";
  $("healthNote").textContent = `${number(summary.active_jobs)} 個工作 · ${number(summary.translation_errors)} 次翻譯錯誤`;
  $("updatedAt").textContent = `最後更新 ${new Date(data.timestamp).toLocaleString("zh-TW")}`;
}

async function poll() {
  if (!state.token && location.hostname !== "127.0.0.1" && location.hostname !== "localhost") { showLogin(); return; }
  try {
    const headers = state.token ? { Authorization: `Bearer ${state.token}` } : {};
    const response = await fetch(API_URL, { headers, cache: "no-store" });
    if (response.status === 401 || response.status === 503) {
      const body = await response.json().catch(() => ({}));
      if (response.status === 401) { state.token = ""; sessionStorage.removeItem(TOKEN_KEY); }
      showLogin(body.detail || "無法連線到 Dashboard");
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    setConnection(false, "連線中斷");
    $("updatedAt").textContent = `更新失敗：${error.message}`;
  } finally {
    clearTimeout(state.timer);
    state.timer = setTimeout(poll, POLL_MS);
  }
}

$("loginForm").addEventListener("submit", event => {
  event.preventDefault();
  state.token = $("tokenInput").value.trim();
  if (state.token) sessionStorage.setItem(TOKEN_KEY, state.token);
  $("loginError").textContent = "";
  poll();
});

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) poll();
});

if (state.token) $("tokenInput").value = state.token;
poll();
