# Aimarumaru provider profiles

影片翻譯頁面提供三種處理方案。三者共用本機 faster-whisper，以及同一條 VAD、
來源正規化、語意翻譯群組、結果驗證與 deterministic display cues 流程；目前只替換
翻譯 provider：

| 方案 | ASR | 翻譯 |
| --- | --- | --- |
| Standard | 本機 faster-whisper | Groq `openai/gpt-oss-120b` |
| Premium | 本機 faster-whisper | Gemini |
| Private | 本機 faster-whisper | 本機 Ollama `qwen3:14b` |

`std`、`pro` 仍分別相容於 `standard`、`premium`。後端會依建立 job 時的
`processing_profile` 強制翻譯路由，後續代理請求無法自行改成其他 provider。
Private 的字幕翻譯也會由 Kotobamaru 停用 Wikidata／Wikipedia 遠端查詢。

本機 ASR 預設使用 faster-whisper `turbo`，保留既有 VAD、逐字時間戳與字幕流程。
若要切回較慢但多語精度較保守的 `large-v3`，可在 `.env` 覆寫：

```dotenv
WHISPER_MODEL_NAME=turbo
# WHISPER_MODEL_NAME=large-v3
```

`start_audioio.sh` 啟動時會先把指定的 faster-whisper 模型快取到 `HF_HOME`；模型
已存在時不會重新下載。若部署環境必須完全離線，可設定：

```dotenv
HF_HOME=/vault/cache/huggingface
WHISPER_MODEL_NAME=turbo
WHISPER_MODEL_DOWNLOAD=0
```

Groq Whisper adapter 目前保留供日後 A/B 測試，但三個正式方案都使用本機
faster-whisper，因此 Aimarumaru 不需要為 ASR 設定 `GROQ_API_KEY`。若日後重新啟用
Groq ASR，可使用：

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
TRANSCRIBE_WORKER_CONCURRENCY=10
```

`PREFETCH_CHUNKS` 不會超過 queue size。若更重視第一段字幕延遲，可設為 `1`；若
Dashboard 的「音訊等待」仍經常大於零，可同時提高兩個值。

### 公平轉錄排程

轉錄佇列預設允許兩個工作同時進入處理，但本機 Whisper 的 GPU 推論由公平排程器
逐 chunk 輪轉。每個 chunk 預設約 30 秒；長影片完成一個 chunk 後會讓出 GPU，下一個
工作即可取得一個 chunk 的機會，避免長影片獨佔整個佇列。字幕翻譯的語意 group 不會
被 30 秒切塊限制，group 仍可跨 chunk 合併。

```dotenv
# 同時執行的轉錄工作數；1 會退回舊的單工作模式
TRANSCRIBE_WORKER_CONCURRENCY=10
```

Dashboard 與 `audioio-access*.jsonl` 會記錄 `scheduler_wait_ms`，可用來確認工作是否
真的交錯取得 Whisper turn；`input_wait_ms` 則仍代表 FFmpeg 音訊供料等待時間。

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
