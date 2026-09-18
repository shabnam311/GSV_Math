import torch
import gc
from collections import Counter, defaultdict
from .clip_alignment import clip_alignment_score
from .symbolic_check import verify_equations
from .owl_grounding import owl_grounding_score
from .answer_extraction import extract_answer, normalize_answer

# ── Training-matched constants ──
# Training used MAX_PIXELS = 156800 (see project_notebooks/Qwen_2.5v_finetuning.ipynb)
# sqrt(156800) ≈ 396, so PIL pre-resize target is 396×396
MAX_PIXELS = 156800
MIN_PIXELS = 3136   # 56*56 — Qwen2.5-VL minimum
PIL_MAX_SIDE = 396  # pre-resize ceiling to match training resolution

SYSTEM_PROMPT = (
    "You are a math problem solver. Look at the image carefully, "
    "read the question, and solve it step by step. "
    "Show your work and end with: The answer is <your answer>."
)


def cisc_generate_and_vote(model, processor, image, question, num_samples=3):
    # ── Pre-resize to match training resolution ──
    if max(image.size) > PIL_MAX_SIDE:
        image.thumbnail((PIL_MAX_SIDE, PIL_MAX_SIDE))

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]}
    ]

    inputs = processor(
        text=[processor.apply_chat_template(messages, add_generation_prompt=True)],
        images=[image],
        return_tensors="pt",
    ).to("cuda")

    # Length of the prompt tokens — used to slice off the echoed prompt from decode
    prompt_len = inputs["input_ids"].shape[1]

    samples = []
    answer_votes = defaultdict(float)

    for i in range(num_samples):
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            max_length=None,
            use_cache=True,
            temperature=0.7,
            do_sample=True
        )

        # Decode only the GENERATED tokens (skip echoed prompt)
        generated_ids = outputs[0][prompt_len:]
        output_text = processor.decode(generated_ids, skip_special_tokens=True)

        # New robust extraction
        extracted_raw = extract_answer(output_text)
        extracted_ans = normalize_answer(extracted_raw)

        # Module 1: OWL-ViT Object Grounding
        owl_score = owl_grounding_score(image, output_text)

        # Module 2: CLIP Semantic Alignment
        clip_score = clip_alignment_score(image, output_text)

        # Module 5: Symbolic Verification
        sympy_passed = verify_equations(output_text)

        # Handle None values (unable to verify) safely without failing open
        safe_owl = owl_score if owl_score is not None else 0.5
        safe_clip = clip_score if clip_score is not None else 0.5

        # Combine confidence
        confidence = (0.5 * safe_owl) + (0.5 * safe_clip)

        # Penalty for failed symbolic check
        if sympy_passed is False:
            confidence *= 0.7

        answer_votes[extracted_ans] += confidence

        samples.append({
            "trace": output_text,
            "answer": extracted_ans,
            "owl_score": owl_score,
            "clip_score": clip_score,
            "sympy_passed": sympy_passed,
            "final_confidence": confidence
        })

        del outputs
        torch.cuda.empty_cache()
        gc.collect()

    best_answer = max(answer_votes, key=answer_votes.get)
    best_sample = next((s for s in samples if s["answer"] == best_answer), samples[0])

    return best_answer, best_sample["trace"], dict(answer_votes), best_sample["clip_score"], best_sample["sympy_passed"], best_sample["owl_score"]
