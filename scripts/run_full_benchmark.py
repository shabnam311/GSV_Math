import os
import json
from datasets import load_dataset
from tqdm.auto import tqdm
import sys

# Add backend to path for importing local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from pipeline.cisc import cisc_generate_and_vote
from pipeline.model_loader import load_models

def main():
    print("Loading MathVista testmini dataset...")
    dataset = load_dataset("AI4Math/MathVista", split="testmini")
    
    results_file = os.path.join(os.path.dirname(__file__), '..', 'eval_splits', 'cisc_owl_gsv_results.jsonl')
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
            
    print(f"Resuming: {len(completed)}/1000 academic evaluations complete.")

    remaining_samples = [s for s in dataset if str(s["pid"]) not in map(str, completed)]

    if not remaining_samples:
        print("Pipeline already complete!")
        return

    # Load model once
    model, processor = load_models()

    with open(results_file, "a") as f:
        for sample in tqdm(remaining_samples):
            pid = sample["pid"]
            
            answer, trace, votes, clip_score, sympy_passed, owl_score = cisc_generate_and_vote(
                model,
                processor,
                sample["decoded_image"], 
                sample["query"], 
                num_samples=3 
            )
            
            result_dict = {
                "pid": pid,
                "ground_truth": sample["answer"],
                "cisc_final_answer": answer,
                "raw_response": trace,
                "vote_distribution": votes,
                "clip_score": clip_score,
                "sympy_passed": sympy_passed,
                "owl_score": owl_score
            }
            
            f.write(json.dumps(result_dict) + "\n")
            f.flush()
            os.fsync(f.fileno())

    print("? Academic CISC Grounding Pipeline Complete!")

if __name__ == "__main__":
    main()
