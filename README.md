# GSV-Math: Grounded Self-Verifying Math VQA

[![Deploy to Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fshabnam311%2FGSV_Math)

GSV-Math is a multi-modal mathematical reasoning and verification pipeline designed to solve and audit complex geometric and diagrammatic problems. It combines a fine-tuned **Qwen2.5-VL-7B-Instruct** vision-language model with a multi-signal **Self-Verification & Anti-Hallucination Pipeline** (OWL-ViT Object Grounding, CLIP Semantic Alignment, and SymPy Symbolic Checking).

---

## 🌟 Key Features

* **Visual Math Backbone:** Qwen2.5-VL-7B-Instruct fine-tuned for geometric and algebraic reasoning via Unsloth.
* **Model Checkpoint:** [Shabuuuuuuuuuuu/GSV-Math-Qwen2.5-VL-7B-Expert](https://huggingface.co/Shabuuuuuuuuuuu/GSV-Math-Qwen2.5-VL-7B-Expert) (GGUF 4-bit available at [Shabuuuuuuuuuuu/GSV-Math-GGUF](https://huggingface.co/Shabuuuuuuuuuuu/GSV-Math-GGUF)).
* **Confidence-Weighted Self-Consistency (CISC):** Samples multiple reasoning paths and weights voting by grounding and alignment scores.
* **Multi-Signal Verification & Anti-Hallucination Guardrails:**
  * **OWL-ViT Visual Grounding:** Uses zero-shot object detection with an extensive geometric vocabulary (`radius`, `diameter`, `hypotenuse`, `sector`, `vertex`, `axis`, etc.) to verify physical diagram references.
  * **CLIP Semantic Alignment:** Measures topic consistency between the image and generated reasoning using smart salient text selection (preserving both prompt context and concluding derivations).
  * **SymPy Symbolic Checking:** Uses `sympy.parsing.sympy_parser` with `convert_xor` to verify internal mathematical and arithmetic consistency, flagging contradictions and division by zero.
  * **Multi-Signal Fusion:** Balanced decision boundary (`0.4 × OWL + 0.6 × CLIP ≥ 0.4`) with a strict SymPy contradiction veto, preventing false alarms on abstract geometric line art.

---

## 📊 Benchmark Results

| Metric | Accuracy |
|---|---|
| Zero-shot Baseline (LLaVA-1.5B) | 16.30% |
| **Fine-Tuned GSV-Math 7B (Mathtestmini)** | **68.30%** |
| Fine-Tuned GSV-Math 7B (MathV360k holdout) | 90.60% |

### Vision-Dependency Score (VDS)
$$\text{VDS} = \frac{\text{Accuracy}_{\text{Visual}} - \text{Accuracy}_{\text{Blind}}}{\text{Accuracy}_{\text{Visual}}}$$
We evaluate VDS to measure genuine visual grounding versus text-prompt memorization. On MathV360K ablation tests, blind accuracy reached 88.20% (McNemar's $p=0.15$), demonstrating that multimodal models frequently lean on multiple-choice linguistic cues — reinforcing the need for our visual grounding and symbolic verification layers.

---

## 🏗️ Architecture & Project Structure

```text
GSV_Math/
├─ backend/
│  ├─ modal_app.py           # Modal Serverless T4 GPU backend (FastAPI)
│  ├─ hf_space/              # Hugging Face CPU Basic backend (llama.cpp GGUF Docker)
│  └─ pipeline/              # Core verification & inference modules
│     ├─ cisc.py             # Confidence-weighted self-consistency voting
│     ├─ owl_grounding.py    # OWL-ViT visual grounding & geometric vocabulary
│     ├─ clip_alignment.py   # CLIP semantic alignment with salient text selection
│     ├─ symbolic_check.py   # SymPy equation verification & contradiction checks
│     ├─ model_loader.py     # 4-bit quantized base model + LoRA adapter loader
│     └─ answer_extraction.py# Robust LaTeX and boxed answer extraction
├─ frontend/                 # Next.js React application (Tailwind CSS)
│  ├─ src/app/page.tsx       # Paper-worksheet UI, diagram preview & metric visualizer
│  └─ package.json
├─ GSV_Math_Demo_Server.ipynb# One-click Google Colab GPU backend (with ngrok tunnel)
├─ project_notebooks/        # Training, fine-tuning, and evaluation notebooks
└─ legacy/                   # Reference baselines
```

---

## 🚀 Deployment & Running

### Option 1: One-Click Google Colab T4 GPU Server (100% Free, No Card Required)
For evaluations, demos, and live testing with free NVIDIA T4 GPU compute:

1. Open **`GSV_Math_Demo_Server.ipynb`** in [Google Colab](https://colab.research.google.com).
2. Set Runtime to **T4 GPU** (**Runtime → Change runtime type → T4 GPU**).
3. Paste your free [ngrok](https://dashboard.ngrok.com) token in **Cell 1**.
4. Run all cells (**Runtime → Run all**).
5. Copy the generated public URL (`https://...ngrok-free.dev`) and set `NEXT_PUBLIC_MODAL_BACKEND_URL` in Vercel.

---

### Option 2: Serverless T4 GPU on Modal
For production-grade serverless deployment that auto-sleeps at $0/sec:

```bash
# 1. Install Modal CLI
pip install modal
modal setup

# 2. Add Secrets in Modal Dashboard (or via CLI)
#    - huggingface-secret (HF_TOKEN)
#    - api-key (API_KEY = "dev-secret-key")

# 3. Deploy
modal deploy backend/modal_app.py
```

---

### Option 3: Deploy Frontend to Vercel
1. Import this repository into [Vercel](https://vercel.com).
2. Set **Root Directory** to `frontend`.
3. Add Environment Variable:
   * `NEXT_PUBLIC_MODAL_BACKEND_URL` = `<YOUR_MODAL_OR_COLAB_URL>`
4. Deploy!

To run the frontend locally:
```bash
cd frontend
npm install
echo "NEXT_PUBLIC_MODAL_BACKEND_URL=https://your-backend-url" > .env.local
npm run dev
```

---

## 🧪 Interactive Math Test Cases

The UI includes 5 dedicated math-diagram benchmarks designed to evaluate different aspects of the pipeline:

| Benchmark | Topic | Focus Area |
|---|---|---|
| **Pythagoras Theorem** | Geometric proof ($a, b, c^2$) | Object grounding & exponential relations |
| **Triangle Area** | Labeled base & height | Element extraction & formula consistency |
| **Circle Area** | Sector decomposition & radius vs. diameter | Adversarial visual grounding trap |
| **Quadratic Formula** | Visual proof by completing the square | Multi-step algebraic parsing |
| **Overlapping Shapes** | Multi-shape Venn & intersection area | Visual claim noise resistance |

---

## 📜 Acknowledgements
* Fine-tuning powered by [Unsloth](https://github.com/unslothai/unsloth)
* Vision-Language backbone by [Qwen Team (Qwen2.5-VL)](https://github.com/QwenLM/Qwen2.5-VL)
* Grounding and verification powered by [Hugging Face Transformers](https://github.com/huggingface/transformers) & [SymPy](https://www.sympy.org/)
