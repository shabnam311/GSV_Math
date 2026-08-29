# Architecture Overview

GSV-Math uses a decoupled, two-service architecture. We explicitly **do not** run the model inference on Vercel, as Vercel serverless functions do not provide GPU access and have strict execution timeouts that would fail on a 7B VLM + 3-5x CISC voting loop.

## 1. Backend: Inference Engine (Modal)
The heavy lifting lives in the `backend/` directory, deployed to **Modal**.

* **Stack:** FastAPI, PyTorch, Unsloth (Qwen2.5-VL), Transformers (OWL-ViT), spaCy.
* **Responsibility:** Loads the model weights into VRAM. Exposes a `/solve` endpoint that takes a base64 image and question, runs the CISC reasoning loop, and returns the final answer, reasoning trace, and confidence scores.
* **Hardware:** NVIDIA T4 GPU (provisioned via `@app.cls(gpu="T4")`).

## 2. Frontend: User Interface (Vercel)
The user interface lives in the `frontend/` directory, deployed to **Vercel**.

* **Stack:** Next.js (React/Tailwind).
* **Responsibility:** Captures user input (image + question), handles base64 encoding, manages the loading/timeout state, and visualizes the results.
* **Hardware:** Vercel Edge Network / Serverless Functions.

## Diagram

```text
+--------------------------+        HTTPS / JSON        +-----------------------------------+
|   Vercel (frontend)      | -------------------------> |  Modal (backend)                  |
|  gsv-math.vercel.app     |<-------------------------- |  <workspace>--gsv-math-solve.run  |
|                          |        answer + trace      |  T4 GPU                           |
|  - Next.js               |                            |  - Qwen2.5-VL-7B-Instruct (4bit)  |
|  - Image upload UI       |                            |  - OWL-ViT grounding              |
|  - Sample problems       |                            |  - spaCy noun extraction          |
|  - Result display        |                            |  - CISC weighted voting (k=3-5)   |
+--------------------------+                            |  - FastAPI /solve + /health       |
                                                        +-----------------------------------+
```
