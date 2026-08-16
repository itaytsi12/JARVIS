# JARVIS training dataset

Canonical runtime data lives in `data/training_dataset.sqlite3`. Raw causal events and evaluated examples are separate tables. Nothing is uploaded and audio capture is off by default.

Configuration: `TRAINING_DATA_ENABLED`, `TRAINING_RAW_RETENTION_DAYS`, `TRAINING_MAX_LOCAL_MB`, `TRAINING_CAPTURE_AUDIO`, `TRAINING_CAPTURE_CODE_CONTEXT`, and `TRAINING_DEDUP_ENABLED`. Set `TRAINING_DATA_ENABLED=false` to disable capture. Raw events expire according to retention after a curated record exists; curated examples are not removed by raw cleanup.

All writes pass through structured and free-text secret sanitization. `.env` mappings, credentials, cookies, authorization headers, private keys, idle microphone audio, repositories, virtual environments and build outputs are not captured. Code context uses source paths and content hashes so repeated excerpts can be content-addressed.

Capture version 1.3 records prepared/terminal action pairs at their real execution boundaries, explicit action timing, bounded per-attempt retry outcomes, route/model provenance, fallback providers, and token counts when an API reports them. Content-bearing user/STT requests, typed payloads, and private message bodies are represented by intent plus bounded metadata rather than copied into request, plan, or tool traces. Older local records are not rewritten automatically; inspection and export sanitize their known typed/message fields at the read boundary. `training_data.stats` reports outcome counts, model usage, storage size, event span, and average bytes per raw event.

Commands:

```
python -m training_data.stats
python -m training_data.validator
python -m training_data.inspect --action open_application --limit 10
python -m training_data.readiness --database data/training_dataset.sqlite3
python -m training_data.exporter --format sft --output exports/jarvis_sft.jsonl
```

Exports: SFT is useful for successful responses, `tools` for tool-use learning, `coding` for patch/evaluation learning, `preferences` for chosen/rejected training, and `raw` for analysis. Grouped `--split` keeps a task/interaction entirely within one split. Future speech-correction training can use transcript/correction metadata; audio remains disabled unless explicitly enabled and is subject to dataset retention.
