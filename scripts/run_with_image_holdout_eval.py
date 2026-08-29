import os
import json
import sys
from tqdm.auto import tqdm
from datasets import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from pipeline.model_loader import load_models
from pipeline.cisc import cisc_generate_and_vote

def main():
    print("Loading MathV360K dataset pool...")
    base_dataset = load_dataset("lmms-lab/LLaVA-OneVision-Data", "GeoQA+(MathV360K)", split="train")
    
    shuffled_dataset = base_dataset.shuffle(seed=42)
    holdout_set = shuffled_dataset.select(range(20000, 20500))
    
    print(f"Loaded {len(holdout_set)} holdout samples for WITH-IMAGE evaluation.")
    
    results_file = os.path.join(os.path.dirname(__file__), '..', 'eval_splits', 'holdout_with_image_results.jsonl')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    completed = set()
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        completed.add(data["pid"])
                    except json.JSONDecodeError:
                        pass
                        
    remaining_samples = [s for s in holdout_set if str(s["id"]) not in map(str, completed)]
    
    if not remaining_samples:
        print("With-image pipeline already complete!")
        return
        
    model, processor = load_models()
    
    with open(results_file, "a") as f:
        for sample in tqdm(remaining_samples):
            pid = sample["id"]
            question = sample["conversations"][0]["value"].replace("<image>\n", "").replace("\n<image>", "")
            image = sample["image"]
            
            # Use CISC for the with-image baseline to match deployment
            answer, trace, votes, clip, sympy, owl = cisc_generate_and_vote(
                model, processor, image, question, num_samples=3
            )
            
            result_dict = {
                "pid": pid,
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
