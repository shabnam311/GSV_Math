# GSV-Math: Grounded Self-Verifying Math VQA

[![Deploy to Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fshabnam311%2FGSV_Math%2Ftree%2Fmain%2Ffrontend)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)

GSV-Math is a vision-language reasoning pipeline that shows its work. It doesn't just return an answer—it reasons over a mathematical diagram, checks its own reasoning using an open-vocabulary object detector (OWL-ViT), and applies Confidence-Weighted Self-Consistency (CISC) to output the most visually-grounded final answer.

## Tech Stack
* **Reasoning Backbone:** Qwen2.5-VL-7B-Instruct (4-bit, via Unsloth)
* **Grounding Module:** OWL-ViT (`google/owlvit-base-patch32`)
* **Text Extraction:** spaCy (`en_core_web_sm`)
* **Voting Strategy:** CISC (Confidence-Weighted Self-Consistency), k=3-5 samples
* **Backend Inference:** FastAPI on Hugging Face Spaces (ZeroGPU)
* **Frontend UI:** Vercel (Static HTML/JS or Next.js)

## Quick Start

### 1. Live Demo
* **Frontend UI:** [https://gsv-math.vercel.app](https://gsv-math.vercel.app) *(Replace with actual Vercel URL)*
* **Backend API (HF Space):** [https://huggingface.co/spaces/username/gsv-math-demo](https://huggingface.co/spaces/username/gsv-math-demo) *(Replace with actual Space URL)*

### 2. GitHub Repository
* **Repository:** [https://github.com/shabnam311/GSV_Math](https://github.com/shabnam311/GSV_Math)

### 3. Run Locally

**Backend:**
\`\`\`bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn api:app --reload --port 7860
\`\`\`

**Frontend:**
Simply open \`frontend/index.html\` in your browser, or deploy to Vercel using the root \`frontend/\` directory.

## Results (MathVista)

*These results reflect the evaluated zero-shot and CISC-grounded accuracy on the MathVista Testmini split.*

| Model / Configuration | Accuracy |
|-----------------------|----------|
| LLaVA-1.5-7B (Baseline)| ~22.3%  |
| Qwen2.5-VL-7B (Zero-shot) | *Evaluating...* |
| Qwen2.5-VL-7B + OWL-ViT (CISC) | *Evaluating...* |

## Known Limitations
* **Zero-Shot Backbone:** Currently running zero-shot Qwen2.5-VL-7B-Instruct. Domain-specific fine-tuning (SFT) has not been applied yet.
* **GPU Quota:** The live demo runs on Hugging Face ZeroGPU (free tier). You may occasionally hit a quota limit or experience cold-start delays (~1-2 minutes) if the Space is idle.
* **OWL-ViT Generalization:** Open-vocabulary object detection works exceptionally well on concrete diagram features, but may struggle with purely abstract/symbolic mathematical annotations (e.g., highly stylized angle markers).

## Documentation
For a deep dive into the reasoning, architecture, and ablation studies, please refer to our [PROJECT_MASTER_PLAN.md](PROJECT_MASTER_PLAN.md).
