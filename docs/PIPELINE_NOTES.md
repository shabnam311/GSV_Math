# GSV-Math Pipeline Notes

1. **Generator**: Qwen2.5-VL-7B-Instruct running in 4-bit mode generates a raw reasoning trace and final answer.
2. **Grounding**: OWL-ViT (google/owlvit-base-patch32) performs zero-shot object detection on the input image using noun chunks extracted by spaCy (en_core_web_sm) from the generator's trace.
3. **Verification**: Symbolic checking (SymPy) and grounding scores are calculated.
4. **CISC Voting**: Confidence-weighted self-consistency picks the best answer from multiple samples.
