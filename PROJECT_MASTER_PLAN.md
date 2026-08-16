# GSV-Math: Grounded Self-Verifying Multimodal Math Reasoning
## Master Implementation Plan & Reference Document — v5 (Latest-Qwen pivot + Batch Full-Benchmark Protocol + Accuracy Stack)
**Last Updated:** 21 July 2026
**Hardware (primary):** Google Colab Free Tier — Tesla T4 (16GB VRAM), ~12GB RAM
**Hardware (secondary):** Kaggle Free — Dual T4 x2 (32GB VRAM), 30 hrs/week
**Persistent storage:** Google Drive, 5TB — used as the "hard drive" the free-tier GPUs don't have
**Legacy hardware:** OMEN Laptop — RTX 4050 (6GB VRAM) — Part A only
**Base Paper:** Math-LLaVA (Shi et al., 2024) — EMNLP 2024 Findings — [arXiv:2406.17294](https://arxiv.org/abs/2406.17294)
**Base Model (v5 decision):** Qwen3.5-9B (primary candidate, newest generation) → Qwen3-VL-8B-Instruct/Thinking (proven strong fallback) → Qwen2.5-VL-7B-Instruct (safety-net fallback, real numbers already on record)

---

## ⚠️ CHANGE LOG (v4 → v5)

1. **Model landscape re-verified (21 July 2026).** Qwen3.5 (released as a *unified* text+vision family, ~March 2026) is now Alibaba's newest generation and, per Qwen's own released benchmarks, **outperforms Qwen3-VL models across reasoning, coding, agents, and visual understanding** at comparable/smaller size. Qwen3.5-9B beats GPT-5-Nano on MMMU-Pro (70.1 vs 57.2) and MathVision (78.9 vs 62.2). This changes our model-selection decision tree — see B.1a and B.2.
2. **Qwen3-VL-8B is no longer "TBD".** Independently-tracked leaderboard data (llm-stats.com, MathVista-Mini, updated July 2026) shows **Qwen3-VL-8B-Thinking scoring 81.4% on MathVista-mini** — dramatically higher than Qwen2.5-VL-7B's 68.2%, and higher than several models 3-4x its size. This is now our confirmed strong fallback, not a guess.
3. **New Part I — Batch-wise Full-Benchmark Execution Protocol.** Answers the direct question: *yes*, we can and should test on the full, fixed 1,000-sample MathVista `testmini` split every time (we already do), and yes, Google Drive's 5TB lets us do this in **resumable batches** so a Colab disconnect never loses progress and a model is downloaded/cached exactly once. This is now written as a concrete, checkpoint-based procedure rather than a general pattern.
4. **New Part J — The Accuracy Stack (research-verified).** A direct answer to "is there really nothing left to try" — there is. This section separates *inference-time* accuracy levers (cheap, no retraining, apply to any stage) from *training-time* accuracy levers (require a training run) and ranks all of them by expected gain vs. implementation cost, all backed by cited literature current to mid-2026.
5. **Fairness clarification formalized (I.1).** Running on a T4 vs the paper's A800s does not make the comparison unfair — same benchmark split, same protocol, same prompt format. The only real caveat is 4-bit/8-bit quantization vs the papers' full precision, which is now explicitly logged as a disclosed methodological note rather than an apology.
6. Parts A, C, D, E (legacy baseline, backup plans, dataset reference, prior-art citations) are carried forward from v4 largely unchanged except where a v5 section supersedes them — noted inline.

---

# PART A — LEGACY BASELINE (LLaVA-1.5 replication, local hardware)
*Status: COMPLETE / SUPERSEDED — kept for disclosure, not the active pipeline*

## A.1 What the Paper Actually Did (Verified from PDF)

The Math-LLaVA paper is a **data-centric** paper. Core contribution is a **data selection + augmentation pipeline** on LLaVA-1.5.

**Their method:**
1. Started with **LLaVA-1.5-13B**.
2. Collected images from 24 public math-VQA datasets.
3. Trained two ViT classifiers (GPT-4V labels) for **clarity** and **comprehension complexity**.
4. Filtered with 2:3:4:1 oversampling ratio across complexity levels.
5. GPT-4V generated 4 augmentation types: **AskImg**, **CompQ**, **RephQ**, **SimpQ**.
6. Produced **360K QA pairs** (MathV360K) from ~40K images.
7. **Full fine-tuning** (NOT QLoRA) on A800 GPUs.

**Their results on MathVista testmini:**

| Model | Overall Accuracy |
|---|---|
| LLaVA-1.5-13B (base) | 27.7% |
| Math-LLaVA-DS (40K filtered) | 38.2% |
| Math-LLaVA (360K augmented) | **46.6%** |

**What they did NOT do:** Chain-of-Thought, visual grounding, self-verification.

## A.2 Our Legacy Results

| Configuration | Overall | FQA | GPS | MWP | TQA | VQA |
|---|---|---|---|---|---|---|
| Zero-Shot LLaVA-1.5-7B ✅ | 22.3% | 22.7% | 14.4% | 10.8% | 33.5% | 33.0% |
| Fine-Tuned Pilot (2K) ✅ | 20.0% | 20.1% | **17.8%** | 9.1% | 24.7% | 29.6% |
| Fine-Tuned Full (20K) | Abandoned — superseded by Qwen pipeline |

**Key finding:** GPS improved +3.4% on tiny 2K pilot → learning signal confirmed.

## A.3 Technical Issues Solved (do not re-debug)

1. C: Drive full → moved HF cache to `D:\huggingface_cache`
2. CPU-only PyTorch → `torch 2.5.1+cu121`
3. `transformers==5.13.1` crash → pinned `4.48.2`
4. Windows access violation → larger pagefile
5. VS Code GPU crash → native PowerShell
6. Token mismatch → `max_length=768`
7. `grad_norm: nan` → `bf16=True`
8. Slow training → lowered LoRA rank, fewer modules
9. Offline hangs → `local_files_only=True`
10. MathVista categories live in `sample["metadata"]["skills"]` (a list)

---

# PART B — MODEL SELECTION (v5, re-verified 21 July 2026)

## B.1 MathVista SOTA Landscape (Verified July 2026)

### Frontier / Closed-Source Models
| Model | MathVista testmini | Notes |
|---|---|---|
| o3 (OpenAI) | ~86.8% | Current leaderboard #1 on full MathVista (llm-stats.com) |
| Kimi K2.5 | ~90.1% | Contested #1 depending on leaderboard/date |
| GPT-5.5 / Gemini 3 Pro | high-80s | Proprietary |

### Open-Source Models Near Our Weight Class (8-10B)
| Model | MathVista-mini | Method | Source |
|---|---|---|---|
| Step3-VL-10B | **84.0%** | Top-ranked fully open model on MathVista overall leaderboard (rank #3 across all models) | llm-stats.com, June 2026 |
| **Qwen3-VL-32B-Thinking** | 85.9% | Reference point — too large for a single free T4 but shows the ceiling of this model family | Qwen3-VL Technical Report, [2511.21631](https://arxiv.org/abs/2511.21631) |
| **Qwen3-VL-8B-Thinking** | **~81.4%** | Confirmed via independent leaderboard tracking (llm-stats.com MathVista-Mini, updated July 2026); no training, out of the box | [llm-stats.com](https://llm-stats.com/benchmarks/mathvista-mini) |
| **Qwen3.5-9B** | Not yet independently benchmarked on MathVista-mini specifically at time of writing; Qwen's own release claims it **"outperforms Qwen3-VL models across reasoning, coding, agents, and visual understanding"** and beats GPT-5-Nano on MathVision (78.9 vs 62.2) and MMMU-Pro (70.1 vs 57.2) — both closely related STEM/visual-reasoning benchmarks | New unified text+vision architecture (Gated DeltaNet + sparse MoE), Apache 2.0 | [Qwen3.5-9B HF](https://huggingface.co/Qwen/Qwen3.5-9B) |
| MathVis-Fine (Qwen2.5-VL-7B + RL) | 77.26% | Vision-dependency reward + progressive training | [2606.17888](https://arxiv.org/abs/2606.17888) |
| Qwen-VL-DP (SFT + GRPO) | ~70.4% | Diverse solving perspectives + RL | [2507.02804](https://arxiv.org/abs/2507.02804) |
| **Qwen2.5-VL-7B-Instruct (zero-shot, published)** | **68.2%** | Out of the box, no training | [Qwen2.5-VL tech report](https://arxiv.org/abs/2502.13923) |
| Math-LLaVA-13B (base paper) | 46.6% | Data augmentation + full FT | [2406.17294](https://arxiv.org/abs/2406.17294) |
| LLaVA-1.5-7B (legacy baseline) | 22.3% | Zero-shot, 4-bit | Our measurement |

### Our Own Measurements So Far
| Run | Overall | Notes |
|---|---|---|
| Qwen2.5-VL-7B-Instruct, zero-shot, `question` field (bug) | 42.40% | Completed, 1000/1000, 5h24m on Colab T4. **Superseded** — missing `query`-field format instructions. |
| Qwen2.5-VL-7B-Instruct, zero-shot, `query` field (fixed) | TBD | Priority rerun — see B.7 step 1 |
| Qwen3-VL-8B-Instruct/Thinking, zero-shot | TBD | Priority — see B.7 step 2 |
| Qwen3.5-9B, zero-shot | TBD | New — see B.7 step 3 |

## B.1a Why Qwen3.5 Changes the Decision (and why we still hedge)

Qwen3.5 is a genuinely different architecture, not just a version bump: it fuses text and vision into **one** unified model from the start (rather than a separate "VL" variant bolted onto a text LLM), uses **Gated DeltaNet** (linear-attention hybrid) for long-context efficiency, and was trained with large-scale RL. Qwen's own comparison tables position it above the equivalent Qwen3-VL size class on STEM/visual-reasoning benchmarks. This is a *believable* claim — DeltaNet-hybrid + native multimodal fusion is a real, published architectural direction, not marketing fluff — but as of this writing we could not find an **independently reported MathVista-mini number** for Qwen3.5-9B specifically (only Qwen's own MathVision/MMMU-Pro figures). Treat the MathVista number as **unverified until we run it ourselves** — which is exactly what Stage 0 is for.

Practically, this means: **run all three candidates' cheap `limit=20` sanity checks before committing to one**, per the decision rule below. This costs almost nothing and removes the guesswork.

## B.2 Model Selection Decision Tree (v5)

```
                        ┌─────────────────────────────┐
                        │  Stage 0-sanity: limit=20    │
                        │  on all 3 candidates         │
                        └──────────────┬───────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
 Qwen3.5-9B                    Qwen3-VL-8B-Thinking            Qwen2.5-VL-7B-Instruct
 loads clean,                  loads clean,                    (known-good baseline,
 generates sensibly?           generates sensibly?              already have 1000-sample
        │                              │                        run on record)
   YES  │  NO                    YES   │  NO
        ▼   └──────────────┐           ▼   └──────────────┐
  ADOPT AS PRIMARY          │     ADOPT AS PRIMARY          │
  for all 5 stages          │     for all 5 stages          │
                             └──────────────┬────────────────┘
                                            ▼
                                   FALL BACK to Qwen2.5-VL-7B
                                   (zero risk — already proven)
```

**Rule:** try Qwen3.5-9B first (newest, best claimed numbers). If it hits friction — `transformers`/Unsloth support still maturing, since it's the newest release — fall to Qwen3-VL-8B-Thinking (now a *confirmed* 81.4% performer, not a guess). If that also hits friction, fall to Qwen2.5-VL-7B (zero risk, already fully proven end-to-end on our exact pipeline).

> [!IMPORTANT]
> Use the **Thinking** variant of Qwen3-VL for accuracy runs, not Instruct — Qwen3-VL's own technical report shows Thinking beating Instruct by several points on MathVista-mini specifically (e.g., 30B-A3B: 81.9 Thinking vs 80.1 Instruct). This reverses our v4 guidance, which favored Instruct for parsing simplicity. The extra chain-of-thought tokens cost more time/compute per sample but are worth it for accuracy — budget generation time accordingly, and keep Instruct as a fast-iteration option during pipeline debugging.
> For Qwen3.5, check at load time whether a `/think` (reasoning) mode toggle is exposed the same way; if so, apply the same principle — reasoning mode for accuracy runs, non-reasoning for fast debug iterations.

### Model IDs to Try (Unsloth-hosted where available)
| Model | ID | Fine-tuning support | VRAM (4-bit) |
|---|---|---|---|
| **Qwen3.5-9B (primary candidate)** | `unsloth/Qwen3.5-9B` / `Qwen/Qwen3.5-9B` | Unsloth `FastVisionModel`/`FastModel` — confirmed in Unsloth's Qwen3.5 fine-tuning guide as of this writing | Dense 9B fits bf16 LoRA on Kaggle dual-T4; 4-bit on single T4 |
| **Qwen3-VL-8B-Thinking (strong fallback)** | `unsloth/Qwen3-VL-8B-Thinking` (verify exact tag; Instruct tag is `unsloth/Qwen3-VL-8B-Instruct`) | Unsloth `FastVisionModel`, confirmed, incl. GRPO/RL support | ~6-8GB |
| **Qwen2.5-VL-7B (safety-net fallback)** | `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit` | Unsloth `FastVisionModel`, most battle-tested path | ~6-8GB |

### Verified Unsloth Loading Pattern — Qwen3.5 (new, primary attempt)
```python
# Per Unsloth's Qwen3.5 fine-tuning guide — verify exact package versions at run time,
# since this is the newest-supported model family as of writing.
!pip install --no-deps unsloth
!pip install --upgrade transformers  # Qwen3.5 requires a very recent transformers build

from unsloth import FastVisionModel  # or FastModel, depending on Unsloth's current Qwen3.5 wrapper — check docs at run time
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3.5-9B",
    load_in_4bit=True,               # switch to load_in_16bit=True on Kaggle dual-T4 for bf16 LoRA (see Part J.5)
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none",
    target_modules="all-linear",   # Unsloth's newer recommended default — verify against current docs
)
```

### Loading Pattern — Qwen3-VL-8B-Thinking (strong, confirmed fallback)
```python
!pip install --no-deps unsloth
!pip install transformers==4.57.1
!pip install --no-deps trl==0.22.2

from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Thinking",   # swap "-Instruct" for faster/cheaper debug runs
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none",
)
```

### Loading Pattern — Qwen2.5-VL-7B (safety-net fallback, unchanged from v4)
```python
from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none",
)
```

## B.3 Hardware Strategy: Colab vs Kaggle (unchanged from v4)

### Google Colab Free Tier
| Feature | Limit |
|---|---|
| GPU | Single T4 (15GB VRAM) — **not guaranteed**, lottery-based |
| Max session | **12 hours** |
| Idle timeout | **~90 minutes** → disconnect |
| Local disk | ~100GB ephemeral — **wiped on disconnect** |
| Dual GPU | ❌ Not available |

### Kaggle Free Tier
| Feature | Limit |
|---|---|
| GPU | **Dual T4 x2** (32GB total) — more reliable |
| Weekly quota | **30 GPU-hours/week** |
| Background runs | ✅ "Commit" and close browser |
| Local disk | 20GB `/kaggle/working` + 73GB `/tmp` |

### Recommended Platform Split
| Task | Platform | Why |
|---|---|---|
| **Fine-tuning (QLoRA or bf16 LoRA)** | **Kaggle** | Dual T4; commit mode; no idle timeout; bf16 needs the 2nd GPU (see J.5) |
| **Full 1000-sample zero/few-shot eval** | **Colab, in batches** (see Part I) | Interactive, but session limits mean batching is mandatory |
| **Grounding module dev** | **Colab** | OWL-ViT + CLIP are small |
| **Rejection-sampling / self-consistency generation** | **Kaggle** | Most GPU-hour-hungry step; needs the weekly quota, not the 12h session cap |
| **Gradio demo** | **Colab** | Easy `share=True` |
| **Model + results storage** | **Google Drive (5TB)** | Download once, batch-checkpoint every run (Part I) |

---

# PART I — BATCH-WISE FULL-BENCHMARK EXECUTION PROTOCOL (NEW in v5)

## I.1 Answering "Is it possible to test on all images fairly, given our hardware?"

**Yes — and you're already doing the part that matters for fairness.** Two things are easy to conflate:

1. **Fairness of the number itself:** comes from running the same, complete, official 1,000-sample MathVista `testmini` split, with the same prompt/answer-extraction protocol the papers use. GPU tier (T4 vs A800) does not change what the *model* outputs for a given prompt — it only changes how *long* it takes to get there. So a full 1,000-sample run on a free T4 is exactly as valid a comparison as the same run on a data-center GPU.
2. **What Drive's 5TB actually buys you:** not fairness — **continuity**. It removes the two things that *would* otherwise force you into unfair shortcuts (testing on a smaller subset, or re-downloading a 16GB model every session and eating into your compute budget). With Drive, you download each model **once**, and you checkpoint **every batch**, so a Colab disconnect costs you a few minutes, not the whole run.

**One honest, disclosed caveat (not a weakness — a standard methodological note):** we run in 4-bit (sometimes bf16 on Kaggle, see J.5) while the original papers mostly report full-precision numbers. State this explicitly in the report as "same benchmark, same protocol, disclosed precision difference" — this is normal practice, not something to apologize for.

## I.2 The Batching Pattern

Rather than one 1,000-sample call that dies if Colab disconnects at sample 850, split the run into fixed-size batches (e.g., 100 samples), write each batch's results to Drive **immediately** on completion, and have the script auto-detect what's already done on startup so it **resumes instead of restarting**.

```python
import json, os, time
from pathlib import Path

DRIVE_ROOT = "/content/drive/MyDrive/gsv_math_results"
RUN_NAME = "qwen35_9b_zeroshot_testmini"     # change per run
BATCH_SIZE = 100                              # 10 batches for the full 1000
RESULTS_PATH = f"{DRIVE_ROOT}/{RUN_NAME}"
os.makedirs(RESULTS_PATH, exist_ok=True)

def load_completed_batches():
    """Scan Drive for batch files already written; return the set of sample indices done."""
    done = set()
    for f in Path(RESULTS_PATH).glob("batch_*.json"):
        with open(f) as fh:
            batch = json.load(fh)
        done.update(item["pid"] for item in batch)
    return done

def save_batch(batch_idx, results):
    """Write one batch to Drive the moment it finishes — never held only in RAM."""
    out_file = f"{RESULTS_PATH}/batch_{batch_idx:04d}.json"
    with open(out_file, "w") as fh:
        json.dump(results, fh)
    print(f"[checkpoint] saved {out_file} ({len(results)} samples)")

# --- main loop ---
completed_pids = load_completed_batches()
print(f"Resuming: {len(completed_pids)}/1000 samples already done.")

remaining = [s for s in dataset if s["pid"] not in completed_pids]

for i in range(0, len(remaining), BATCH_SIZE):
    batch = remaining[i : i + BATCH_SIZE]
    batch_results = []
    for sample in batch:
        # ... run inference on `sample`, using sample["query"] (NOT sample["question"] — see A.3/B.1 bug note) ...
        batch_results.append({"pid": sample["pid"], "prediction": pred, "gt": sample["answer"]})
    save_batch(batch_idx=(len(completed_pids) // BATCH_SIZE) + (i // BATCH_SIZE), results=batch_results)
```

**Why this specific design:**
- **Resumability is index-based, not count-based.** Scanning for which `pid`s are already done (not just "how many batches exist") means a partially-written batch from a mid-batch crash doesn't silently corrupt the resume logic.
- **Batch size of 100 = ~10 checkpoints for the full run.** Small enough that losing one batch to a disconnect costs minutes, large enough that Drive I/O overhead stays negligible.
- **Same pattern reused for every stage** (zero-shot, few-shot, fine-tuned eval, grounded eval, VDS blank-image pass) — one script, swap the `RUN_NAME` and the model/config block.

## I.3 One-Time Model Caching (unchanged principle from v4, now explicit per-model)

```python
# START of every session — cache-or-copy pattern:
from google.colab import drive
drive.mount('/content/drive')

MODEL_CACHE = "/content/drive/MyDrive/model_cache/qwen35-9b"  # one folder per candidate model
LOCAL_PATH  = "/content/qwen35-9b"

import os
if os.path.exists(MODEL_CACHE):
    print("Found cached model on Drive — copying to local SSD (fast).")
    os.system(f"cp -r {MODEL_CACHE} {LOCAL_PATH}")
else:
    print("No cache found — will download once, then save to Drive at end of session.")
    # ... load via from_pretrained(), which downloads to local HF cache ...
    # ... at the end: os.system(f"cp -r {LOCAL_PATH} {MODEL_CACHE}") ...
```

This means: **download each of the 3 candidate models exactly once**, total, across the whole project — not once per session.

## I.4 Aggregation Script (run after all batches for a stage complete)

```python
import json, glob

all_results = []
for f in sorted(glob.glob(f"{RESULTS_PATH}/batch_*.json")):
    with open(f) as fh:
        all_results.extend(json.load(fh))

assert len(all_results) == 1000, f"Incomplete run: {len(all_results)}/1000"

correct = sum(1 for r in all_results if normalize(r["prediction"]) == normalize(r["gt"]))
print(f"Overall accuracy: {correct/len(all_results)*100:.2f}%")

# Save final consolidated result + per-category breakdown to Drive, matching B.8's target table format
with open(f"{RESULTS_PATH}/FINAL_results.json", "w") as fh:
    json.dump({"overall_acc": correct/len(all_results), "n": len(all_results), "results": all_results}, fh)
```

## I.5 Estimated Runtime Budget (for planning session counts, not a promise)

| Run type | Est. time on single T4 | Notes |
|---|---|---|
| Zero-shot, Instruct variant, full 1000 | 4-6 hours | Comparable to our recorded Qwen2.5-VL-7B run (5h24m) |
| Zero-shot, **Thinking** variant, full 1000 | 6-10+ hours | Extended CoT generation is slower per sample — budget 2-3 Colab sessions, using the batch-resume pattern to span them |
| Few-shot (2-4 exemplars), full 1000 | Similar to zero-shot + small overhead | Longer prompts, same generation length |
| Self-consistency (k=5 samples/question), full 1000 | ~5x the zero-shot time | This is the accuracy-lever from Part J — budget it as its own multi-session block |

---

# PART J — THE ACCURACY STACK (NEW in v5, research-verified)

## J.1 Framing

The honest starting point: **your zero-shot backbone accuracy is not "your" number** — it's the model vendor's. What *is* legitimately yours to improve is the **delta on top of it**. This section catalogs every credible, currently-documented technique for growing that delta, split into two buckets:

- **Inference-time levers** — no retraining required, apply on top of *any* stage (zero-shot, few-shot, or fine-tuned), cheap to test, and can be layered.
- **Training-time levers** — require a training run, larger effort, larger potential payoff.

All are cited; all are labeled honestly by how novel vs. "well-established and just not yet in your pipeline" they are — same standard as Part B.6's novelty framing.

## J.2 Inference-Time Levers (cheapest, apply first)

### J.2.1 Self-Consistency / Majority-Vote Decoding — *highest confidence, lowest effort*
Sample the model **k times** (k=5-10) at temperature ~0.7-1.0 instead of once greedily, extract the final answer from each, and take the majority vote. This is one of the best-documented test-time-scaling techniques in the literature — <cite index="43-1">generating multiple diverse outputs and using voting to select the best answer has produced 5-25% accuracy improvement on reasoning tasks</cite>, and it's already a standard baseline used specifically <cite index="48-1">in multimodal reasoning tasks such as MMMU, where it often bridges a significant portion of the gap between base models and oracle performance</cite>.

- **Cost:** k× inference compute at eval time only — zero training. On a 1000-sample benchmark with k=5, budget ~5x the zero-shot runtime (use Part I's batching, this is exactly what it's for).
- **Where it fits your pipeline:** apply it *before* Module 6 (grounding-based Best-of-N rerank) — they compose. Self-consistency picks the most-agreed-upon *answer*; Module 6 can be layered on top to break ties using visual grounding score instead of frequency alone when votes are split.
- **Prior art / novelty status:** well-established (Wang et al. 2023, self-consistency decoding) — present as "applying a documented technique," not a contribution, same framing as Module 6.

### J.2.2 Confidence-Weighted Self-Consistency (CISC-style) — *modest additional gain over plain majority vote*
Plain majority voting treats every sampled answer equally. A documented refinement has the model (or a lightweight scorer) assign a confidence to each sampled reasoning trace and use **weighted** voting instead of raw counts — <cite index="52-1">weighted majority voting, which assigns a confidence value to each candidate answer and chooses the answer with the largest accumulated score, tends to be more accurate on a wide range of popular benchmarks</cite> than unweighted majority vote.
- **Cheap version for your pipeline:** you already compute a CLIP grounding-alignment score per candidate (Module 2) — reuse *that* as the per-candidate weight instead of building a separate confidence model. This is a natural, low-cost extension of infrastructure you're already planning to build, not new infrastructure.
- **Prior art:** Taubenfeld et al. 2025 (CISC); framing as adaptation, not invention.

### J.2.3 Best-of-N with Early Stopping — *efficiency lever, not pure accuracy, but frees up budget for more samples elsewhere*
Rather than always sampling a fixed k, stop early once confidence stabilizes. Documented result: <cite index="51-1">applying confidence-based Early Stopping to Best-of-N improves MathQA accuracy from 81.0 to 83.6 with a sample budget of 16 responses</cite> — i.e. smarter allocation of the same compute, not just "sample more." Given your free-tier GPU-hour constraints, this matters: it lets you spend saved compute on more *questions* getting self-consistency treatment rather than fewer questions getting excessive over-sampling.

### J.2.4 Fixed Answer-Extraction Prompt Discipline — *already partially fixed, worth re-stating as a lever*
You already found and are fixing the `query`-vs-`question` field bug (B.1). This is worth generalizing: MathVista's own scoring is sensitive to answer-format compliance (multiple-choice letter vs. free-form number vs. exact string). A consistent, explicit "Answer with only the final value/letter, no explanation" instruction (or "put your final answer in \boxed{}" for CoT/Thinking models, matching how Qwen3.5's own benchmarks were scored) measurably affects parseable-answer rate, independent of the model's actual reasoning quality. Treat this as a **zero-cost accuracy floor-raiser** to apply before anything else.

## J.3 Training-Time Levers

### J.3.1 Rejection-Sampling Fine-Tuning / STaR-style Bootstrapping — *carried forward from the v4 discussion, now formalized*
Already agreed in this project: SFT pass → sample k candidate solutions per training question → keep only ones reaching the correct answer → fine-tune again on the expanded, self-verified set. Zero external API cost; bootstraps from your own model. One iteration only — literature shows diminishing returns after 2-4 rounds. This remains the single most substantial training-time addition in the whole plan; nothing in this new research changes that ranking.

### J.3.2 Curriculum / Progressive-Difficulty Training Order — *new to v5, cheap to add*
Rather than shuffling all 20K training samples randomly, **order them from easier to harder** within each epoch (e.g., by question length, number of reasoning steps implied, or your existing complexity-classifier signal from the base paper's own filtering pipeline in A.1). MathVis-Fine's own 77.26% result explicitly credits "progressive training" alongside its RL reward — this is the SFT-only, non-RL analogue of that idea, and it costs only a re-sort of your existing dataset, not new infrastructure.
- **Implementation note:** you already have a natural difficulty proxy sitting unused — MathV360K's source sub-datasets differ in typical difficulty (IconQA/TabMWP tend easier than Geometry3K/UniGeo geometric proofs). A coarse curriculum (sub-dataset order) is a one-line change to your data loader.

### J.3.3 bf16/16-bit LoRA on Kaggle Dual-T4 instead of 4-bit QLoRA — *removes a documented accuracy tax*
This directly answers "can we use our 5TB/dual-GPU advantage for something other than storage." 4-bit NF4 quantization introduces a small but real accuracy tax vs. full or half precision. Kaggle's dual T4 (32GB combined) has enough headroom, via `device_map="auto"` sharding, to run **bf16 LoRA** (not QLoRA) on an 8-9B model — roughly 16-18GB of bf16 weights split ~8-9GB per GPU, leaving room for activations and gradients. This was already flagged in the prior turn of this project and is reconfirmed here as still the right call: **not novel, but a real, hardware-specific accuracy improvement you're positioned to use and most published 7-8B-class papers (running on single high-VRAM GPUs) don't bother distinguishing in their own ablations.**

### J.3.4 Data Augmentation via Multi-Crop / Multi-Resolution Training Views — *new to v5*
For geometry-heavy sub-datasets (GeoQA+, UniGeo, Geometry3K) specifically, include a second training view of each image at a different crop/resolution (e.g., a tight crop around the diagram vs. the full original) mapped to the same QA pair. This is a lightweight way to make the model less sensitive to exact framing — directly relevant to your GPS-category weakness (A.2 legacy results: GPS was the lowest-scoring category at 14.4%). Cost: doubles the effective image-loading time for the augmented subset only (recommend applying to ~30% of geometry samples, not the full 20K, to keep training time bounded).

### J.3.5 Optional Stretch — Short GRPO/RL Pass on Top of SFT+RFT
Unchanged from v4's stretch-goal framing: Unsloth explicitly supports Vision RL/GRPO now, and this is the recipe OpenVLThinker and Qwen-VL-DP used to reach 69-70%. Sequence it **last**, after SFT (J.3.1's base pass) and RFT (J.3.1's bootstrap pass) are both working — RL on top of a weak base is a well-documented way to waste GPU-hours for little gain, RL on top of an already-improved SFT+RFT model is where the literature shows it paying off.

## J.4 Recommended Stacking Order (cheapest/highest-confidence first)

```
1. Fixed prompt/answer-extraction discipline         (J.2.4)  — apply immediately, free
2. Self-consistency majority vote, k=5                (J.2.1)  — apply at every eval stage
3. Curriculum ordering for the SFT pass                (J.3.2)  — one-line data loader change
4. bf16 LoRA on Kaggle instead of 4-bit QLoRA          (J.3.3)  — swap config on Kaggle only
5. RFT / STaR bootstrap round                          (J.3.1)  — as already planned
6. Geometry multi-crop augmentation                    (J.3.4)  — targeted at your weakest category
7. Grounding + CLIP verification + symbolic check      (B.5 Modules 1,2,5) — as already planned
8. CLIP-weighted self-consistency (CISC-style)         (J.2.2)  — reuses Module 2's output, cheap add-on
9. Best-of-N grounding rerank                          (B.5 Module 6) — as already planned
10. (Stretch) short GRPO/RL pass                        (J.3.5)  — only if time remains
```

Each of steps 2, 3, 4, 5, 6 should get its **own line in the results table** (B.8-style), not be silently bundled — that's what turns "we tried some things" into a real ablation study, and it's more report-worthy than the accuracy number alone.

## J.5 Updated B.8-style Target Table (v5, with accuracy-stack rows)

| Stage | ALL (target) | Notes |
|---|---|---|
| Qwen2.5-VL-7B zero-shot (fixed `query` prompt) | ~68% (published) | Priority rerun to confirm on our exact eval harness |
| Qwen3-VL-8B-Thinking zero-shot | ~81% (leaderboard-reported) | Priority — confirm independently |
| Qwen3.5-9B zero-shot | TBD — unverified on MathVista specifically | Priority — first real data point |
| + Self-consistency (k=5) on chosen zero-shot model | +2-10pt over above (literature range 5-25% is task-dependent; expect the lower end on an already-strong backbone) | J.2.1 |
| Stage 1: Fine-tuned (SFT, curriculum-ordered, bf16 LoRA) | Backbone-dependent — aim to close remaining gap to backbone's own published ceiling | J.3.2 + J.3.3 |
| Stage 1.5: + RFT/STaR bootstrap round | +modest, literature shows diminishing but real returns | J.3.1 |
| Stage 2: + Grounding + Verification + Symbolic + Best-of-N + CISC-weighting | +2-3pt over Stage 1.5 (original B.9 estimate, now with CISC as an extra lever) | B.5 + J.2.2 |

---

# PART C — BACKUP PLANS (unchanged from v4)

| Plan | Trigger | Risk |
|---|---|---|
| **B: Prompting + Self-Consistency** | Stage 2 ≈ Stage 0/1 | LOW — note this plan is now partially *absorbed* into the main pipeline via J.2.1, not just a fallback |
| **D: Distillation from GPT-4o-mini** | Need stronger reasoning traces | MEDIUM |
| **E: Narrow to Geometry (GPS) only** | Need faster, focused result | LOW |
| **F: Hallucination Reduction framing** | Accuracy delta is small | LOW (**RECOMMENDED DEFAULT**) |
| **G: Ensemble Voting** | Zero-shot + fine-tuned disagree | LOW — note overlap with J.2.1/J.2.2; if you build self-consistency, ensemble voting across *models* (not just samples) is a near-zero-additional-cost extension |

---

# PART D — DATASETS & MODELS REFERENCE (unchanged from v4)

## Datasets
| Dataset | Size | Role | Link |
|---|---|---|---|
| MathVista testmini | 1,000 | Evaluation ONLY | [HuggingFace](https://huggingface.co/datasets/AI4Math/MathVista) |
| MathV360K (embedded) | ~83K (5 sub-datasets) | Training | [lmms-lab/LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data) |

### MathV360K Sub-datasets
```python
geoqa    = load_dataset("lmms-lab/LLaVA-OneVision-Data", "GeoQA+(MathV360K)",     split="train")
unigeo   = load_dataset("lmms-lab/LLaVA-OneVision-Data", "UniGeo(MathV360K)",     split="train")
geometry = load_dataset("lmms-lab/LLaVA-OneVision-Data", "Geometry3K(MathV360K)", split="train")
iconqa   = load_dataset("lmms-lab/LLaVA-OneVision-Data", "IconQA(MathV360K)",     split="train")
tabmwp   = load_dataset("lmms-lab/LLaVA-OneVision-Data", "TabMWP(MathV360K)",     split="train")
```

## File Locations (Drive layout, extended for v5 multi-model caching)

### Google Drive (persistent)
| What | Path |
|---|---|
| Model weights (per candidate) | `/content/drive/MyDrive/model_cache/<model-name>/` |
| LoRA checkpoints | `/content/drive/MyDrive/gsv_math_checkpoints/<run-name>/` |
| Batched eval results | `/content/drive/MyDrive/gsv_math_results/<run-name>/batch_XXXX.json` |
| Consolidated final results | `/content/drive/MyDrive/gsv_math_results/<run-name>/FINAL_results.json` |

---

# PART E — KEY REFERENCES (extended for v5)

## Core References (unchanged from v4)
| Paper | Venue | Status |
|---|---|---|
| Shi et al. — **Math-LLaVA** | EMNLP 2024, [2406.17294](https://arxiv.org/abs/2406.17294) | ✅ Published |
| Lu et al. — **MathVista** | ICLR 2024, [2310.02255](https://arxiv.org/abs/2310.02255) | ✅ Published |
| Bai et al. — **Qwen2.5-VL** | 2025, [2502.13923](https://arxiv.org/abs/2502.13923) | ✅ Published |
| Qwen Team — **Qwen3-VL Technical Report** | 2025, [2511.21631](https://arxiv.org/abs/2511.21631) | ✅ Published |
| Dettmers et al. — **QLoRA** | NeurIPS 2023 | ✅ Published |

## New v5 References — Accuracy Stack
| Paper | Method | Relevance |
|---|---|---|
| Wang et al. 2023 — Self-Consistency Decoding | Multi-sample + majority vote | Core technique behind J.2.1 |
| Taubenfeld et al. 2025 — CISC | Confidence-weighted majority voting | Basis for J.2.2 |
| "Efficient Test-Time Scaling via Self-Calibration", [2503.00031](https://arxiv.org/abs/2503.00031) | Confidence-based early stopping for Best-of-N | Basis for J.2.3 |
| "Boosting Self-Consistency with Ranking" (RISC), [2606.05054](https://arxiv.org/abs/2606.05054) | Ranking-augmented self-consistency, beats plain majority vote with fewer samples | Cite as a possible upgrade path if plain self-consistency underperforms |
| MathVis-Fine, [2606.17888](https://arxiv.org/abs/2606.17888) | Vision-dependency reward + progressive (curriculum) training | Basis for J.3.2's curriculum framing |

---

# PART H — PRODUCTIZATION (unchanged from v4)

*No changes in v5. Refer to v4 document for the full two-tier Gradio/Hugging Face Spaces plan.*

---

# PART K — REVISED PRIORITY ORDER IF TIME RUNS SHORT (v5)

1. **Batch-protocol full 1000-sample zero-shot on the winning model from the decision tree** (Part I + B.2) — non-negotiable
2. **Apply J.2.1 (self-consistency) + J.2.4 (prompt discipline) on top of that zero-shot number** — cheap, no training, immediate credible accuracy gain to report
3. **Tier 1 static demo** (Part H.3) — guarantees a presentable "product" even if nothing else lands in time
4. **SFT with curriculum ordering + bf16 LoRA on Kaggle** (J.3.2 + J.3.3) — your actual fine-tuning contribution
5. **RFT/STaR bootstrap round** (J.3.1)
6. **Grounding + Verification + Symbolic check** (Module 5) — strongest remaining novelty claim
7. **CISC-weighted self-consistency + Best-of-N rerank** (J.2.2 + Module 6) — accuracy levers, layer on top once the above works
8. **Stage-wise VDS measurement** (Module 7) — diagnostic, report-worthy even if accuracy gains are modest
9. **Geometry multi-crop augmentation** (J.3.4) — targeted fix for your weakest category if time allows
10. **Tier 2 live inference deployment / GRPO stretch pass** — cut first if time runs short

---

*This document supersedes v4 as the single source of truth for the GSV-Math project.*
*Part A is frozen. Parts C, D, H carry forward from v4 unchanged except where noted.*
*All model numbers, benchmarks, and technique citations verified via web research through 21 July 2026.*
