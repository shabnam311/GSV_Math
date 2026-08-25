import torch
import gc
from collections import Counter

def cisc_generate_and_vote(model, processor, image, question, num_samples=3):
    """
    Runs Confidence-Weighted Self-Consistency (CISC) by sampling multiple
    reasoning traces, extracting final answers, and returning the majority vote.
    """
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]}
    ]
    
    inputs = processor(text=[processor.apply_chat_template(messages, add_generation_prompt=True)], 
                       images=[image], 
                       return_tensors="pt").to("cuda")
    
    samples = []
    
    for i in range(num_samples):
        # Generate with temperature for diversity in CISC
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            max_length=None,  # Suppresses the max_length warning!
            use_cache=True,
            temperature=0.7,
            do_sample=True
        )
        
        output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Simple extraction for now: assume answer is the last word or after "answer is"
        # In a full deployment, this calls verification.py to extract regex and calculate grounding
        ans_start = output_text.lower().rfind("answer is")
        if ans_start != -1:
            extracted_ans = output_text[ans_start + 9:].strip().strip(".")
        else:
            extracted_ans = output_text.split()[-1].strip(".")
            
        samples.append({
            "trace": output_text,
            "answer": extracted_ans
        })
        
        # Free memory aggressively inside loop
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        
    # Voting logic
    answers = [s["answer"] for s in samples]
    counts = Counter(answers)
    best_answer = counts.most_common(1)[0][0]
    
    # Find the trace that matches the best answer
    best_trace = next((s["trace"] for s in samples if s["answer"] == best_answer), samples[0]["trace"])
    
    return best_answer, best_trace, dict(counts)
