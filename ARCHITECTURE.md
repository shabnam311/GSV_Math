# GSV-Math Architecture & System Design

GSV-Math uses a decoupled, multi-modal verification architecture to combine vision-language reasoning with symbolic and object-level verification guardrails.

---

## High-Level System Architecture

```text
+-------------------------------------------------------------------------+
|                           Vercel Frontend UI                            |
|                          (Next.js / TypeScript)                         |
|  - Paper-worksheet visual interface & LaTeX equation renderer           |
|  - 5 Interactive visual math benchmark diagrams                         |
|  - Multi-signal breakdown (OWL-ViT, CLIP, SymPy, Hallucination Verdict) |
+-------------------------------------------------------------------------+
                                    │
                                    │ HTTPS (JSON + Base64 Image)
                                    ▼
+-------------------------------------------------------------------------+
|                  Inference & Verification Backend                       |
|          (Modal Serverless T4 GPU  OR  Google Colab T4 GPU)             |
+-------------------------------------------------------------------------+
       │                             │                            │
       ▼                             ▼                            ▼
┌──────────────────┐       ┌──────────────────┐         ┌──────────────────┐
│  Reasoning Engine│       │  Visual Grounding│         │  Symbolic Audit  │
│  Qwen2.5-VL-7B   │       │  OWL-ViT + CLIP  │         │  SymPy Parser    │
│  LoRA Expert     │       │  Salient Window  │         │  convert_xor     │
└──────────────────┘       └──────────────────┘         └──────────────────┘
       │                             │                            │
       └─────────────────────────────┼────────────────────────────┘
                                     ▼
                   ┌───────────────────────────────────┐
                   │ CISC Fusion & Hallucination Gate │
                   │  - 0.4*OWL + 0.6*CLIP >= 0.4      │
                   │  - Strict SymPy contradiction     │
                   │    override (veto)                │
                   └───────────────────────────────────┘
```

---

## Core Pipeline Components

### 1. Reasoning Backbone
* **Model:** `Qwen/Qwen2.5-VL-7B-Instruct` loaded in 4-bit (`BitsAndBytesConfig` / NF4).
* **LoRA Adapter:** `Shabuuuuuuuuuuu/GSV-Math-Qwen2.5-VL-7B-Expert` fine-tuned on diagrammatic mathematics.
* **CISC Voting:** Generates multiple reasoning traces at temperature 0.7 and weights candidate answers using alignment and grounding confidences.

### 2. Multi-Signal Verification Modules (`backend/pipeline/`)

| Module | Source File | Description & Guardrails |
|---|---|---|
| **Visual Grounding** | `owl_grounding.py` | `google/owlvit-base-patch32` zero-shot object detector filtered through an expanded geometric vocabulary (`radius`, `hypotenuse`, `sector`, `angle`, `vertex`, etc.). Fails safely into `None` if claims cannot be detected. |
| **Semantic Alignment** | `clip_alignment.py` | `openai/clip-vit-base-patch32` cross-modal topic alignment. Employs smart salient text selection to retain both the question context and the concluding mathematical derivation without token overflow. |
| **Symbolic Verification** | `symbolic_check.py` | `sympy.parsing.sympy_parser` with `convert_xor` to verify exact arithmetic and equation consistency (`simplify(lhs - rhs) == 0`). Properly handles free variables (`x+5=12` → `None`) and explicitly flags `ZeroDivisionError` as arithmetic hallucination. |

### 3. Fusion & Hallucination Decision Boundary
$$\text{Score}_{\text{combined}} = 0.4 \cdot \text{Score}_{\text{OWL}} + 0.6 \cdot \text{Score}_{\text{CLIP}}$$
$$\text{Hallucination Detected} = \begin{cases} \text{YES}, & \text{if } \text{SymPy} = \text{FAILED} \\ \text{YES}, & \text{if } \text{Score}_{\text{combined}} < 0.4 \\ \text{NO}, & \text{otherwise} \end{cases}$$

---

## Deployment Topologies

1. **Modal Serverless GPU (`backend/modal_app.py`):**
   * Production serverless backend provisioned with NVIDIA T4 GPUs.
   * Container caches model layers in persistent volumes and scales down to 0 instances when idle ($0/sec idle cost).

2. **Google Colab GPU Server (`GSV_Math_Demo_Server.ipynb`):**
   * Free NVIDIA T4 GPU serving via `transformers` 4-bit quantization + `pyngrok` secure tunnel.
   * Ideal for live demonstrations, evaluations, and project defense sessions with 0 cloud subscription cost.

3. **CPU-Basic GGUF (`backend/hf_space/`):**
   * 4-bit GGUF quantization powered by `llama.cpp` (`llama-server`) for CPU-only execution environments.
