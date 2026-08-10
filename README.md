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
```

`PREFETCH_CHUNKS` 不會超過 queue size。若更重視第一段字幕延遲，可設為 `1`；若
Dashboard 的「音訊等待」仍經常大於零，可同時提高兩個值。

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
