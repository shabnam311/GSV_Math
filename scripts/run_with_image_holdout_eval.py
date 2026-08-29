import os
import json
import sys
import hashlib
import io
from tqdm.auto import tqdm
from datasets import load_dataset
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from pipeline.model_loader import load_models
from pipeline.cisc import cisc_generate_and_vote

def sample_identity(example):
    human_msg = example["conversations"][0]["value"]
    text_part = human_msg.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "").strip()
    img = example["image"]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    h = hashlib.sha256()
    h.update(text_part.encode("utf-8"))
    h.update(img_bytes)
    return h.hexdigest()

def main():
    print("Loading MathV360K dataset pool...")
    base_dataset = load_dataset("lmms-lab/LLaVA-OneVision-Data", "GeoQA+(MathV360K)", split="train")
    
    hashes_file = os.path.join(os.path.dirname(__file__), '..', 'eval_splits', 'train_sample_hashes.json')
    if not os.path.exists(hashes_file):
        raise FileNotFoundError(f"{hashes_file} not found. Please place your training hashes here to prevent leakage.")
        
    with open(hashes_file, "r") as f:
        train_hashes = set(json.load(f))
        
    print(f"Loaded {len(train_hashes)} training hashes. Filtering leakage...")
    
    clean_indices = []
    for idx, ex in enumerate(tqdm(base_dataset, desc="Hashing GeoQA+ samples")):
        try:
            if sample_identity(ex) not in train_hashes:
                clean_indices.append(idx)
        except Exception:
            continue
            
    print(f"Found {len(clean_indices)} unseen samples out of {len(base_dataset)}.")
    
    # Take the exact same 500 samples the notebook uses
    holdout_indices = clean_indices[-500:]
    holdout_set = base_dataset.select(holdout_indices)
    
    holdout_hashes = {sample_identity(ex) for ex in holdout_set}
    assert holdout_hashes.isdisjoint(train_hashes), "Leakage check failed — do not trust this eval."
    
    print(f"Loaded {len(holdout_set)} holdout samples for WITH-IMAGE evaluation. Zero overlap verified.")
    
    results_file = os.path.join(os.path.dirname(__file__), '..', 'eval_splits', 'holdout_with_image_results.jsonl')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    completed = set()
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        completed.add(data.get("hash_id") or data.get("pid") or data.get("id"))
                    except json.JSONDecodeError:
                        pass
                        
    # Filter based on hash_id since original dataset has weird IDs sometimes
    remaining_samples = []
    for s in holdout_set:
        s_hash = sample_identity(s)
        if s_hash not in completed and str(s.get("id")) not in completed:
            remaining_samples.append(s)
    
    if not remaining_samples:
        print("With-image pipeline already complete!")
        return
        
    model, processor = load_models()
    
    with open(results_file, "a") as f:
        for sample in tqdm(remaining_samples):
            pid = sample.get("id")
            s_hash = sample_identity(sample)
            question = sample["conversations"][0]["value"].replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            image = sample["image"]
            
            # Use CISC for the with-image baseline to match deployment
            answer, trace, votes, clip, sympy, owl = cisc_generate_and_vote(
                model, processor, image, question, num_samples=3
            )
            
            result_dict = {
                "pid": pid,
                "hash_id": s_hash,
                "ground_truth": sample["conversations"][1]["value"],
                "cisc_final_answer": answer,
                "raw_response": trace
            }
            
            f.write(json.dumps(result_dict) + "\n")
            f.flush()
            os.fsync(f.fileno())
            
    print("? With-Image Eval Complete!")

if __name__ == "__main__":
    main()
