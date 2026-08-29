import json
import os
import random

def generate_splits():
    # Simulate loading Math360k dataset ids
    print("Generating deterministic train/test splits to prevent contamination...")
    os.makedirs('eval_splits', exist_ok=True)
    
    # Normally we would load the actual dataset and map PIDs.
    # We will generate a frozen array of IDs for 20000 training samples
    # and 500 holdout samples that have ZERO overlap.
    
    total_samples = 360000 
    all_ids = list(range(total_samples))
    
    random.seed(42)
    random.shuffle(all_ids)
    
    train_ids = all_ids[:20000]
    holdout_ids = all_ids[-500:]
    
    assert len(set(train_ids) & set(holdout_ids)) == 0, "Leakage detected!"
    
    with open('eval_splits/train_ids.json', 'w') as f:
        json.dump(train_ids, f)
        
    with open('eval_splits/holdout_ids.json', 'w') as f:
        json.dump(holdout_ids, f)
        
    print(f"Saved {len(train_ids)} train IDs and {len(holdout_ids)} holdout IDs.")

if __name__ == '__main__':
    generate_splits()
