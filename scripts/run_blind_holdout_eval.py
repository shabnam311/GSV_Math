import os
import json
import sys
from tqdm.auto import tqdm
from datasets import load_dataset

# Add backend to path for importing local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from pipeline.model_loader import load_models
from pipeline.answer_extraction import extract_answer, normalize_answer

def text_only_generate(model, processor, question, num_samples=1):
    messages = [
        {"role": "user", "content": [
            # CRITICAL: No image token here. True blind eval.
            {"type": "text", "text": question}
        ]}
    ]
    
    # Process text-only
    text_input = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_input], return_tensors="pt").to("cuda")
    
    # Generate
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256,
        do_sample=False # Greedy for deterministic blind baseline
    )
    
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    extracted_raw = extract_answer(output_text)
    extracted_ans = normalize_answer(extracted_raw)
    
    return extracted_ans, output_text

def main():
    print("Loading MathV360K dataset pool...")
    base_dataset = load_dataset("lmms-lab/LLaVA-OneVision-Data", "GeoQA+(MathV360K)", split="train")
    
    shuffled_dataset = base_dataset.shuffle(seed=42)
    holdout_set = shuffled_dataset.select(range(20000, 20500))
    
    print(f"Loaded {len(holdout_set)} holdout samples for BLIND evaluation.")
    
    results_file = os.path.join(os.path.dirname(__file__), '..', 'eval_splits', 'holdout_blind_results.jsonl')
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
        print("Blind pipeline already complete!")
        return
        
    model, processor = load_models()
    
    with open(results_file, "a") as f:
        for sample in tqdm(remaining_samples):
            pid = sample["id"]
            question = sample["conversations"][0]["value"].replace("<image>\n", "").replace("\n<image>", "")
            
            answer, trace = text_only_generate(model, processor, question)
            
            result_dict = {
                "pid": pid,
                "ground_truth": sample["conversations"][1]["value"],
                "cisc_final_answer": answer,
                "raw_response": trace
            }
            
            f.write(json.dumps(result_dict) + "\n")
            f.flush()
            os.fsync(f.fileno())
            
    print("? Blind Eval Complete!")

if __name__ == "__main__":
    main()
