# GSV-Math CPU Backend

A permanently free, always-on backend for the GSV-Math visual math solver.

Runs **Qwen2.5-VL-7B + LoRA** on CPU using `llama.cpp` for fast inference,
hosted on Hugging Face Spaces (CPU Basic: 2 vCPU, 16GB RAM, $0/month).

## Architecture

```
Vercel Frontend → HF Space (FastAPI :7860) → llama-server :8081 (C++ / GGUF)
```

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds llama.cpp from source, installs Python deps |
| `start.sh` | Downloads GGUF models from HF, starts llama-server + FastAPI |
| `app.py` | FastAPI proxy matching the original Modal API contract |
| `requirements.txt` | Python dependencies (no PyTorch needed!) |
