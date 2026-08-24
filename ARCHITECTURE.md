# Architecture Overview

GSV-Math uses a decoupled, two-service architecture. We explicitly **do not** run the model inference on Vercel, as Vercel serverless functions do not provide GPU access and have strict execution timeouts that would fail on a 7B VLM + 3-5x CISC voting loop.

## 1. Backend: Inference Engine (Hugging Face Spaces)
The heavy lifting lives in the `backend/` directory, deployed to a Hugging Face Space using the **Docker SDK** and **ZeroGPU**.

* **Stack:** FastAPI, PyTorch, Unsloth (Qwen2.5-VL), Transformers (OWL-ViT), spaCy.
* **Responsibility:** Loads the model weights into VRAM. Exposes a `/predict` endpoint that takes a base64 image and question, runs the CISC reasoning loop, and returns the final answer, reasoning trace, and confidence scores.
* **Hardware:** H200 (via ZeroGPU dynamic assignment).

## 2. Frontend: User Interface (Vercel)
The user interface lives in the `frontend/` directory, deployed to **Vercel**.

* **Stack:** Static HTML/CSS/JS (or Next.js).
* **Responsibility:** Captures user input (image + question), handles base64 encoding, manages the loading/timeout state, and visualizes the results.
* **Hardware:** Vercel Edge Network / Serverless Functions.

## Diagram

```text
┌─────────────────────────┐        HTTPS / JSON        ┌──────────────────────────────────┐
│   Vercel (frontend)     │ ───────────────────────────▶│  Hugging Face Space (backend)    │
│  gsv-math.vercel.app    │◀─────────────────────────── │  <user>-gsv-math-demo.hf.space   │
│                          │        answer + trace       │  ZeroGPU (H200, on-demand)       │
│  - Static HTML/JS/CSS   │                              │  - Qwen2.5-VL-7B-Instruct (4bit) │
│  - Image upload UI      │                              │  - OWL-ViT grounding             │
│  - Sample problems      │                              │  - spaCy noun extraction         │
│  - Result display       │                              │  - CISC weighted voting (k=3-5)  │
└─────────────────────────┘                              │  - FastAPI /predict + /health    │
                                                       └──────────────────────────────────┘
```
