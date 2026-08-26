import torch
import gc
from collections import Counter, defaultdict
from .clip_alignment import clip_alignment_score
from .symbolic_check import verify_equations

def cisc_generate_and_vote(model, processor, image, question, num_samples=3):
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
    answer_votes = defaultdict(float)
    
    for i in range(num_samples):
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            max_length=None,
            use_cache=True,
            temperature=0.7,
            do_sample=True
        )
        
        output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        ans_start = output_text.lower().rfind("answer is")
        if ans_start != -1:
            extracted_ans = output_text[ans_start + 9:].strip().strip(".")
        else:
            extracted_ans = output_text.split()[-1].strip(".")
            
        # Module 2: CLIP Semantic Alignment
        clip_score = clip_alignment_score(image, output_text)
        
        # Module 5: Symbolic Verification
        sympy_passed = verify_equations(output_text)
        
        # Combine confidence
        # Baseline confidence is 1.0 (since OWL-ViT isn't fully wired in the deployed Modal yet)
        # We blend it with CLIP: 0.5 * 1.0 + 0.5 * clip_score
        confidence = 0.5 + (0.5 * clip_score)
        
        # Penalty for failed symbolic check
        if sympy_passed is False:
            confidence *= 0.7 
            
        answer_votes[extracted_ans] += confidence
            
        samples.append({
            "trace": output_text,
            "answer": extracted_ans,
            "clip_score": clip_score,
            "sympy_passed": sympy_passed,
            "final_confidence": confidence
        })
        
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        
    best_answer = max(answer_votes, key=answer_votes.get)
    best_sample = next((s for s in samples if s["answer"] == best_answer), samples[0])
    
    # We return the best trace, the vote dict, and the new module signals
    return best_answer, best_sample["trace"], dict(answer_votes), best_sample["clip_score"], best_sample["sympy_passed"]
