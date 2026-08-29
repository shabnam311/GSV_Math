import json
import os
import sys

def mcnemar_test(b, c):
    """
    Computes McNemar's test for paired nominal data.
    b: Number of questions where With-Image was WRONG but Blind was RIGHT.
    c: Number of questions where With-Image was RIGHT but Blind was WRONG.
    """
    if b + c == 0:
        return 1.0  # Perfect agreement, no significance
    chi_square = ((abs(b - c) - 1) ** 2) / (b + c)
    # Using scipy for p-value if available
    try:
        from scipy.stats import chi2
        p_value = chi2.sf(chi_square, 1)
        return p_value
    except ImportError:
        print("Install scipy for exact p-value (pip install scipy). Returning raw Chi-Square.")
        return chi_square

def run_vds_evaluation(with_image_results_file, blind_results_file):
    print(f"Loading With-Image results from {with_image_results_file}")
    with open(with_image_results_file, 'r') as f:
        with_img = [json.loads(line) for line in f if line.strip()]
        
    print(f"Loading Blind results from {blind_results_file}")
    with open(blind_results_file, 'r') as f:
        blind = [json.loads(line) for line in f if line.strip()]
        
    # Index by PID
    with_img_dict = {str(item["pid"]): item for item in with_img}
    blind_dict = {str(item["pid"]): item for item in blind}
    
    common_pids = set(with_img_dict.keys()).intersection(set(blind_dict.keys()))
    print(f"Found {len(common_pids)} overlapping evaluations.")
    
    both_correct = 0
    both_wrong = 0
    img_right_blind_wrong = 0 # 'c' in McNemar
    img_wrong_blind_right = 0 # 'b' in McNemar
    
    for pid in common_pids:
        img_ans = str(with_img_dict[pid].get("cisc_final_answer", "")).lower()
        blind_ans = str(blind_dict[pid].get("cisc_final_answer", "")).lower()
        gt = str(with_img_dict[pid].get("ground_truth", "")).lower()
        
        img_correct = (img_ans == gt)
        blind_correct = (blind_ans == gt)
        
        if img_correct and blind_correct:
            both_correct += 1
        elif not img_correct and not blind_correct:
            both_wrong += 1
        elif img_correct and not blind_correct:
            img_right_blind_wrong += 1
        elif not img_correct and blind_correct:
            img_wrong_blind_right += 1
            
    total = both_correct + both_wrong + img_right_blind_wrong + img_wrong_blind_right
    if total == 0:
        print("No valid comparisons found.")
        return
        
    print("\n=== Vision-Dependency Score (VDS) Report ===")
    print(f"Total Evaluated: {total}")
    print(f"With-Image Accuracy: {((both_correct + img_right_blind_wrong) / total) * 100:.2f}%")
    print(f"Blind Accuracy:      {((both_correct + img_wrong_blind_right) / total) * 100:.2f}%")
    
    print("\n--- Contingency Table ---")
    print(f"Both Correct: {both_correct}")
    print(f"Both Wrong:   {both_wrong}")
    print(f"Image ONLY Correct: {img_right_blind_wrong}")
    print(f"Blind ONLY Correct: {img_wrong_blind_right}")
    
    p_val = mcnemar_test(img_wrong_blind_right, img_right_blind_wrong)
    print(f"\nMcNemar's Test p-value: {p_val}")
    if p_val > 0.05:
        print("Result: NOT STATISTICALLY SIGNIFICANT. The model relies almost entirely on text/multiple-choice options.")
    else:
        print("Result: STATISTICALLY SIGNIFICANT. The model uses the visual modality.")

if __name__ == '__main__':
    print("Run this script using the generated JSONL outputs from the pipeline:")
    print("python scripts/run_vds_blind_eval.py <with_image.jsonl> <blind.jsonl>")
    if len(sys.argv) == 3:
        run_vds_evaluation(sys.argv[1], sys.argv[2])
