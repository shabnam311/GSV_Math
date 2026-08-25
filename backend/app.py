import gradio as gr
import spaces
import torch
import os
from pipeline.model_loader import load_models
from pipeline.cisc import cisc_generate_and_vote

# Initialize globally so cold-start is paid once per Space wake-up
print("Booting GSV-Math Backend...")
model, processor = load_models()

@spaces.GPU
def process_math_query(image, question):
    if image is None or not question.strip():
        return "Error: Please provide both an image and a question.", "", {}
        
    # VRAM-safe image resizing guard to protect ZeroGPU pool
    if max(image.size) > 768:
        image.thumbnail((768, 768))
        
    try:
        # Run CISC voting loop
        best_answer, best_trace, vote_dist = cisc_generate_and_vote(
            model, processor, image, question, num_samples=3
        )
        return best_answer, best_trace, vote_dist
    except Exception as e:
        return f"Inference Error: {str(e)}", "", {}

with gr.Blocks(title="GSV-Math", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ?? GSV-Math: Grounded Self-Verifying Math VQA")
    gr.Markdown("**Research Demo:** Evaluates multimodal math reasoning using Qwen2.5-VL with Confidence-Weighted Self-Consistency (CISC).")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Upload Math Diagram")
            query_in = gr.Textbox(label="Question", placeholder="E.g., What is the value of x in the diagram?")
            submit_btn = gr.Button("Solve", variant="primary")
            
        with gr.Column(scale=1):
            ans_out = gr.Textbox(label="Final Answer (Majority Vote)")
            trace_out = gr.Textbox(label="Reasoning Trace (Winning Sample)", lines=8)
            dist_out = gr.JSON(label="Vote Distribution (CISC)")
            
    submit_btn.click(
        fn=process_math_query,
        inputs=[image_in, query_in],
        outputs=[ans_out, trace_out, dist_out]
    )

if __name__ == "__main__":
    demo.queue().launch()
