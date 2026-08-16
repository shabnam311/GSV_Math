import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"
import gradio as gr
from PIL import ImageDraw
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import importlib.util
import sys

# Load novel modules dynamically
try:
    from importlib.machinery import SourceFileLoader
    novel_module = SourceFileLoader("novel", "03_grounding_verification.py").load_module()
except Exception as e:
    print(f"Error loading novel modules: {e}")

print("Loading GSV-Math Model (LLaVA-1.5-7B 4-bit)...")
model_id = "llava-hf/llava-1.5-7b-hf"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

llava_processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    quantization_config=quantization_config,
    device_map={"": 0},
    local_files_only=True
)

print("Initializing Novel Modules (OWL-ViT & CLIP)...")
novel_verifier = novel_module.NovelModules()

def predict(image, question):
    if image is None: return None, "Please upload an image.", ""
    
    # 1. Run LLaVA
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = llava_processor(text=prompt, images=image, return_tensors="pt").to(llava_model.device, torch.float16)
    
    with torch.no_grad():
        outputs = llava_model.generate(**inputs, max_new_tokens=128)
    
    raw_answer = llava_processor.decode(outputs[0], skip_special_tokens=True)
    answer = raw_answer.split("ASSISTANT:")[-1].strip()
    
    # 2. Grounding (OWL-ViT)
    key_phrase = novel_module.extract_key_phrase(answer)
    bbox, grounding_score = novel_verifier.ground_image(image, key_phrase)
    
    # 3. Verification (CLIP)
    similarity, confidence_label = novel_verifier.verify_grounding(image, bbox, key_phrase)
    
    # 4. Draw Bounding Box
    img_with_box = image.copy()
    if bbox:
        draw = ImageDraw.Draw(img_with_box)
        draw.rectangle(bbox, outline="red", width=3)
    
    reasoning = f"Extracted Visual Element: '{key_phrase}'\n{confidence_label}"
    return img_with_box, answer, reasoning

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Math Problem Image"),
        gr.Textbox(label="Your Question", placeholder="e.g., What is the area of the triangle?")
    ],
    outputs=[
        gr.Image(label="Image with Grounded Region Highlighted"),
        gr.Textbox(label="LLaVA's Answer"),
        gr.Textbox(label="Visual Grounding + Confidence")
    ],
    title="GSV-Math: Grounded Self-Verifying Math VQA",
    description="Upload a math problem image and ask a question. The AI answers and verifies exactly which part of the diagram it used."
)

if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch(share=False) # Change to True for public link
