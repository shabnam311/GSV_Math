import os
import json
from datasets import load_dataset
from tqdm.auto import tqdm
import sys

# Add backend to path for importing local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from cisc import cisc_generate_and_vote
import config

def main():
    print("Loading MathVista testmini dataset...")
    dataset = load_dataset("AI4Math/MathVista", split="testmini")
    
    results_file = os.path.join(config.RESULTS_DIR, "cisc_owl_gsv_results.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    results = []
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
            
    completed = {r["pid"] for r in results}
    print(f"Resuming: {len(completed)}/1000 academic evaluations complete.")

    remaining_samples = [s for s in dataset if s["pid"] not in completed]

    for sample in tqdm(remaining_samples):
        pid = sample["pid"]
        
        best_ans, best_reasoning, votes, _ = cisc_generate_and_vote(
            sample["decoded_image"], 
            sample["query"], 
            num_samples=3 
        )
        
        results.append({
            "pid": pid,
            "ground_truth": sample["answer"],
            "cisc_final_answer": best_ans,
            "raw_response": best_reasoning,
            "vote_distribution": votes
        })
        
        if len(results) % 5 == 0:
            with open(results_file, "w") as f: 
                json.dump(results, f, indent=4)
                
    with open(results_file, "w") as f: 
        json.dump(results, f, indent=4)

    print("✅ Academic CISC Grounding Pipeline Complete!")

if __name__ == "__main__":
    main()
