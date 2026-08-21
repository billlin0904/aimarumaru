# Aimarumaru provider profiles

影片翻譯頁面提供三種處理方案。三者共用同一個 ASR provider，以及同一條 VAD、
來源正規化、語意翻譯群組、結果驗證與 deterministic display cues 流程；處理方案
只替換翻譯 provider：

| 方案 | ASR | 翻譯 |
| --- | --- | --- |
| Standard | `AUDIOIO_ASR_PROVIDER` | Groq `openai/gpt-oss-120b` |
| Premium | `AUDIOIO_ASR_PROVIDER` | Gemini |
| Private | `AUDIOIO_ASR_PROVIDER` | 本機 Ollama `qwen3:14b` |

`std`、`pro` 仍分別相容於 `standard`、`premium`。後端會依建立 job 時的
`processing_profile` 強制翻譯路由，後續代理請求無法自行改成其他 provider。
Private 的字幕翻譯也會由 Kotobamaru 停用 Wikidata／Wikipedia 遠端查詢。

ASR 可在本機 faster-whisper、Cloudflare Workers AI、Together Realtime 與保留的
Groq adapter 間切換：

```dotenv
AUDIOIO_ASR_PROVIDER=local
# AUDIOIO_ASR_PROVIDER=cloudflare
# AUDIOIO_ASR_PROVIDER=together
# AUDIOIO_ASR_PROVIDER=groq
```

provider 都會正規化成相同的 segment 與語言資訊，因此 SSE、SRT 及後續翻譯不需要
分別處理。Cloudflare 不提供逐字 confidence；Together Realtime 不提供 segment／word
時間碼，因此會以每次 commit 的音訊區間建立粗粒度時間軸，逐字時間會省略。

本機 ASR 使用 faster-whisper `turbo`，保留既有 VAD、逐字時間戳與字幕流程。若要
切回較慢但多語精度較保守的 `large-v3`，可在 `.env` 覆寫：

```dotenv
WHISPER_MODEL_NAME=turbo
# WHISPER_MODEL_NAME=large-v3
```

若 YouTube 開始要求 Proof of Origin（PO）Token 才能取得音訊串流，建議使用
bgutil provider 自動針對每支影片產生 token（yt-dlp 官方建議方式）。先在 Aimarumaru
主機啟動 provider：Linux 使用 `./start_pot_provider.sh`；Windows 可直接執行
`start_pot_provider.bat`，或使用
`powershell -ExecutionPolicy Bypass -File .\start_pot_provider.ps1`。再在 `.env` 加入：

```dotenv
YOUTUBE_POT_PROVIDER_URL=http://127.0.0.1:4416
```

接著重新啟動 Aimarumaru。provider 與 `bgutil-ytdlp-pot-provider` 外掛會由
`requirements.txt` 安裝；Aimarumaru 會把 provider URL 傳給 yt-dlp，不需要讀取、
保存或貼上使用者瀏覽器的登入 token。

`YOUTUBE_PO_TOKEN` 僅保留為臨時手動 fallback，且必須只存在伺服器的 `.env`，
不要放進 PureText 前端、瀏覽器擴充套件或 Git：

```dotenv
# token 必須保留 client/context 前綴，例如 web_music.gvs+<token>
YOUTUBE_PO_TOKEN=web_music.gvs+請填入你的Token
```

舊的 `PO_TOKEN_VALUE` 仍可使用，惟 `YOUTUBE_PO_TOKEN` 為新部署建議名稱。修改後需
重新啟動 Aimarumaru；程式會在下載音軌、建立串流與取得 YouTube 資訊時傳給 yt-dlp。

`start_audioio.sh` 啟動時會先把指定的 faster-whisper 模型快取到 `HF_HOME`；模型
已存在時不會重新下載。若部署環境必須完全離線，可設定：

```dotenv
HF_HOME=/vault/cache/huggingface
WHISPER_MODEL_NAME=turbo
WHISPER_MODEL_DOWNLOAD=0
```

Cloudflare Workers AI 使用 Base64 WAV JSON 呼叫 REST API，回傳內容會由 adapter
轉成 faster-whisper 相容格式：

```dotenv
AUDIOIO_ASR_PROVIDER=cloudflare
CLOUDFLARE_ACCOUNT_ID=請填入帳號ID
CLOUDFLARE_API_TOKEN=請填入新Token
CLOUDFLARE_ASR_MODEL=@cf/openai/whisper-large-v3-turbo
CLOUDFLARE_ASR_BASE_URL=https://api.cloudflare.com/client/v4
CLOUDFLARE_ASR_TIMEOUT_SECONDS=120
CLOUDFLARE_ASR_MAX_RETRIES=2
CLOUDFLARE_ASR_FALLBACK_WAIT_SECONDS=5
CLOUDFLARE_ASR_MAX_WAIT_SECONDS=30
CLOUDFLARE_ASR_MAX_IN_FLIGHT=50
CLOUDFLARE_ASR_RATE_LIMIT_PER_MINUTE=600
CLOUDFLARE_ASR_QUEUE_MAX_SIZE=1000
TRANSCRIBE_REMOTE_WORKER_CONCURRENCY=50
```

影片仍由 FFmpeg 逐段產生音訊，但所有 Cloudflare chunk 會進入同一個 process-wide
公平佇列。排程器一次只保留每個活躍工作的下一段，輪流服務不同 job，並同時限制
in-flight 與 60 秒滑動視窗內的請求數。`TRANSCRIBE_REMOTE_WORKER_CONCURRENCY`
控制本機同時維持的遠端影片／音訊串流數，應依 CPU、網路與 FFmpeg 負載調整；它與
Cloudflare API 的 `CLOUDFLARE_ASR_MAX_IN_FLIGHT` 是兩個獨立限制。

這些限制以單一 Python process 計算。若以多個 Uvicorn worker 啟動，每個 process
都會各自建立一套排程器，需按 worker 數量降低 rate limit，或改用外部共享佇列。

Groq Whisper adapter 仍保留供 A/B 測試。若要啟用 Groq ASR，可使用：

```dotenv
GROQ_API_KEY=gsk_請填入自己的金鑰
GROQ_ASR_BASE_URL=https://api.groq.com/openai/v1
GROQ_ASR_MODEL=whisper-large-v3
GROQ_ASR_TIMEOUT_SECONDS=120
GROQ_ASR_MAX_RETRIES=2
GROQ_ASR_FALLBACK_WAIT_SECONDS=10
GROQ_ASR_MAX_WAIT_SECONDS=30
GROQ_ASR_MIN_REQUEST_INTERVAL_SECONDS=3.1
```

Groq ASR 使用 OpenAI-compatible `/audio/transcriptions` multipart API。音訊仍先由
本機 Silero VAD 判斷；完全無語音的 chunk 不會送出，回傳 segment 也必須與語音
區間重疊。預設最短請求間隔 3.1 秒，用來避免免費方案超過 20 RPM。

Together 的原文辨識、語言偵測與講者辨識全部使用 Whisper Large v3。一般轉譯走
Realtime WebSocket；開啟「講者辨識（Batch 測試）」後，先由 Batch HTTP 的
`diarize=true` 取得講者時間區段，再把每個講者區段送入同一個 Whisper Realtime
模型辨識原文。這條流程完全使用 Together，不依賴 Cloudflare。Realtime 音訊使用
16 kHz mono PCM frame；`delta` 只做即時預覽，`completed` 才會成為正式字幕並進入
翻譯：

```dotenv
AUDIOIO_ASR_PROVIDER=together
TOGETHER_API_KEY=請填入自己的金鑰
TOGETHER_ASR_BASE_URL=https://api.together.ai/v1
TOGETHER_ASR_MODEL=openai/whisper-large-v3
TOGETHER_REALTIME_URL=wss://api.together.ai/v1/realtime
TOGETHER_ASR_TIMEOUT_SECONDS=120
TOGETHER_ASR_MAX_RETRIES=2
TOGETHER_REALTIME_MAX_RETRIES=1
TOGETHER_REALTIME_CHUNK_SECONDS=10
TOGETHER_REALTIME_FRAME_BYTES=4096
TOGETHER_BATCH_CHUNK_SECONDS=600
TOGETHER_MULTILINGUAL_SOURCE_CHUNK_SECONDS=30
TOGETHER_DIARIZATION_MIN_SPEAKERS=1
TOGETHER_DIARIZATION_MAX_SPEAKERS=5
```

自動語言使用 `language=auto`；手動選擇語言時，Realtime WebSocket 會傳入指定語言。
Realtime 連線中斷只會重送尚未收到 `completed` 的目前 chunk，不會重跑整支影片。

講者辨識會自動略過 YouTube 內建字幕。長影片會依
`TOGETHER_BATCH_CHUNK_SECONDS` 分批處理；同一批內的講者編號穩定，但跨批次時服務端
可能重新分配講者編號，因此需要全片一致身分的場景宜先使用較大的安全批次。自動語言
講者模式的時間精度為講者區段，不提供推測性的逐字時間碼。

YouTube 音訊預設限制在 96 kbps 以內，降低串流供料等待；若來源沒有合適格式，會
依序退回 128 kbps 與最佳可用音軌：

```dotenv
YOUTUBE_AUDIO_FORMAT=bestaudio[abr<=96]/bestaudio[abr<=128]/bestaudio
```

## Resumable video upload

影片上傳頁使用 20 MiB 分塊，每個分塊獨立通過 Cloudflare，連線中斷後重新選擇
同一個檔案即可從伺服器已收到的分塊續傳。所有分塊完成後才在伺服器合併並建立
Whisper 工作；頁面會顯示每個檔案與整批上傳進度。暫存分塊預設保留兩小時：

```dotenv
VIDEO_UPLOAD_CHUNK_BYTES=20971520
VIDEO_UPLOAD_SESSION_TTL_SECONDS=7200
```

單一影片仍受 `VIDEO_UPLOAD_MAX_BYTES` 限制，預設為 2 GiB。舊版單次 multipart
上傳 API 保留相容性，新版頁面使用 `/api/video-upload/sessions/*`。

Kotobamaru 需另外設定 Groq 與 Gemini 翻譯金鑰，Aimarumaru 透過既有
`TRANSLATE_API_BASE` 代理翻譯請求。

## Operations Dashboard

`/dashboard` 顯示 NVIDIA GPU、VRAM、轉錄佇列、Whisper chunk 耗時、音訊等待、
模型推論、事件送出、處理倍率、翻譯延遲／錯誤及 provider 回傳的估算費用。
API 只輸出遙測，不輸出影片網址、字幕內容、翻譯權杖或 API 金鑰。

本機 Whisper 預設先準備兩個 PCM chunk，再開始第一批推論，降低 FFmpeg 供料速度
短暫波動造成 GPU 間歇性閒置。代價是第一段字幕會晚幾秒出現；可依磁碟與來源網路
調整：

```dotenv
YOUTUBE_WHISPER_STREAM_QUEUE_SIZE=2
YOUTUBE_WHISPER_STREAM_PREFETCH_CHUNKS=2
TRANSCRIBE_WORKER_CONCURRENCY=2
TRANSCRIBE_REMOTE_WORKER_CONCURRENCY=50
TRANSCRIBE_QUEUE_MAX_SIZE=200
```

`PREFETCH_CHUNKS` 不會超過 queue size。若更重視第一段字幕延遲，可設為 `1`；若
Dashboard 的「音訊等待」仍經常大於零，可同時提高兩個值。

### 公平轉錄排程

本機與遠端工作使用不同的併發槽。本機 Whisper 的 GPU 推論由公平排程器逐 chunk
輪轉；Cloudflare ASR 則由共用 API 佇列輪轉並行。每個 chunk 預設約 30 秒，長影片
完成一段後不會獨佔下一個 API 機會。字幕翻譯的語意 group 不受 30 秒切塊限制，
group 仍可跨 chunk 合併。

```dotenv
# 同時執行的轉錄工作數；1 會退回舊的單工作模式
TRANSCRIBE_WORKER_CONCURRENCY=2
# 同時維持的 Cloudflare/Groq/Together 影片音訊串流數
TRANSCRIBE_REMOTE_WORKER_CONCURRENCY=50
TRANSCRIBE_QUEUE_MAX_SIZE=200
```

Dashboard 與 `audioio-access*.jsonl` 會記錄 `scheduler_wait_ms`，可用來確認工作是否
真的交錯取得 Whisper turn 或 Cloudflare slot；`input_wait_ms` 則仍代表 FFmpeg
音訊供料等待時間。`GET /api/transcribe-queue/status` 另會回傳本機／遠端活躍工作數，
以及 Cloudflare ASR 的 pending、in-flight、近 60 秒請求量與限制值。

`TRANSCRIBE_QUEUE_MAX_SIZE` 是等待中的工作上限，預設為 200；超過上限的新請求會
立即收到 503「轉譯佇列已滿，請稍後再試」，不會建立一個永久停在 `QUEUED` 的工作。

### English/Japanese/Korean ASR A/B benchmark

`benchmark_artifact_asr.py` 會用 `F:\kotobamaru\artifacts` 的英、日、韓三支影片，
直接走正式環境的 FFmpeg 串流切塊、VAD、beam size 與逐字時間戳流程，依序比較
`large-v3` 與 `turbo`。預設各測前 60 秒，並輸出 SRT、逐段 JSON 與 Markdown 報告：

```powershell
python benchmark_artifact_asr.py
python benchmark_artifact_asr.py --duration 0
```

第二個指令會跑完整影片。報告的 transcript agreement 是兩模型的一致程度，不是
人工正確率；實際品質仍應對照輸出的 SRT 與影片內容。

本機未設定 Token 時只允許 localhost 讀取；雲端部署必須在 `.env` 設定：

```dotenv
AUDIOIO_DASHBOARD_TOKEN=請使用足夠長的隨機字串
```

可用 Linux 產生 Token：

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

設定後重新啟動 Aimarumaru，再開啟 `https://你的網域/dashboard`。

測試：

```powershell
C:\Users\User\anaconda3\envs\aimarumaru\python.exe -m unittest discover -s tests -p "test_*.py" -v
node --test tests/test_subtitle_display_cues.js tests/test_translation_usage.js
```
