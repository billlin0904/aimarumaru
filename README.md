# Aimarumaru provider profiles

影片翻譯頁面提供三種處理方案。三者共用同一條 VAD、來源正規化、語意翻譯群組、
結果驗證與 deterministic display cues 流程，只替換 ASR 與翻譯 provider：

| 方案 | ASR | 翻譯 |
| --- | --- | --- |
| Standard | Groq `whisper-large-v3` | Groq `openai/gpt-oss-120b` |
| Premium | Groq `whisper-large-v3` | Gemini |
| Private | 本機 faster-whisper | 本機 Ollama `qwen3:14b` |

`std`、`pro` 仍分別相容於 `standard`、`premium`。後端會依建立 job 時的
`processing_profile` 強制翻譯路由，後續代理請求無法自行改成其他 provider。
Private 的字幕翻譯也會由 Kotobamaru 停用 Wikidata／Wikipedia 遠端查詢。

Standard／Premium 需要在啟動 Aimarumaru 前設定：

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

測試：

```powershell
C:\Users\User\anaconda3\envs\aimarumaru\python.exe -m unittest discover -s tests -p "test_*.py" -v
node --test tests/test_subtitle_display_cues.js tests/test_translation_usage.js
```
