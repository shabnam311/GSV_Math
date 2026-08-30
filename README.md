# GSV-Math: Grounded Self-Verifying Math VQA

[![Deploy to Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fshabnam311%2FGSV_Math)

GSV-Math is a vision-language reasoning pipeline built to solve complex geometric and mathematical diagrams. It utilizes a fine-tuned **Qwen2.5-VL-7B-Instruct** model, paired with Confidence-Weighted Self-Consistency (CISC) to vote on multiple semantic reasoning paths and arrive at the most robust final answer.

## Tech Stack
* **Reasoning Backbone:** Qwen2.5-VL-7B-Instruct (Fine-tuned with Unsloth)
* **Model Checkpoint:** [Shabuuuuuuuuuuu/GSV-Math-Qwen2.5-VL-7B-Expert](https://huggingface.co/Shabuuuuuuuuuuu/GSV-Math-Qwen2.5-VL-7B-Expert)
* **Voting Strategy:** CISC (Confidence-Weighted Self-Consistency)
* **Backend Inference:** Serverless GPU via [Modal](https://modal.com/) (FastAPI + T4 GPU)
* **Frontend UI:** Next.js + Tailwind CSS, hosted on [Vercel](https://vercel.com/)

## Results

**Vision-Dependency Score (VDS)**
To prevent the model from simply guessing answers based on textual patterns without looking at the diagram, we measure the Vision-Dependency Score (VDS).
VDS = (Accuracy_Visual - Accuracy_Blind) / Accuracy_Visual
A higher VDS indicates the model is genuinely using the image to solve the problem rather than hallucinating from the text prompt.
The model was fine-tuned and evaluated against mathematical reasoning benchmarks. 

| Metric | Accuracy |
|--------|----------|
| Zero-shot Baseline | 16.30% |
| **Fine-Tuned (Tested on Mathtestmini)** | **68.30%** |
| Fine-Tuned (Tested on Math360k) | 90.60% |


## Known Limitations & VDS Findings
While the model achieves 90.60% accuracy on the MathV360K holdout set when provided with the image, rigorous ablation testing reveals a critical limitation in its visual grounding. 

We computed the **Vision-Dependency Score (VDS)** by evaluating the model on the exact same questions with the images entirely removed (blind evaluation). 
* **With-Image Accuracy:** 90.60%
* **Blind Accuracy:** 88.20%

A McNemar's statistical significance test yielded **p = 0.15**, meaning the difference between seeing the image and being blind is *not statistically significant*. The model is heavily relying on the text of the multiple-choice questions rather than genuinely grounding its reasoning in the visual geometry.

## Live Demo
The application features a custom, lightweight "paper worksheet" UI that interacts directly with the serverless GPU backend.

- **Frontend:** [gsv-math.vercel.app](https://gsv-math-git-main-shabnam311s-projects.vercel.app) 
- **Backend:** Hosted serverlessly on Modal.

## Project Structure
```text
GSV_Math/
├─ backend/            # Backend deployment options
│  ├─ modal_app.py    # Modal serverless GPU backend (FastAPI)
│  ├─ hf_space/       # Hugging Face CPU Basic backend (llama.cpp Docker)
│  └─ pipeline/       # CISC voting, prompt formatting, model inference
├─ frontend/           # Next.js React web application
│  ├─ src/app/        # Page routing, React components, and CSS
│  └─ package.json
├─ project_notebooks/  # Training, fine-tuning, and evaluation scripts (Unsloth)
└─ legacy/             # Original LLaVA baseline (superseded, kept for reference)
```

## Deployment / Running Locally

### 1. Deploy the Backend (Modal)
You will need a free [Modal](https://modal.com) account and a Hugging Face token.

```bash
pip install modal
modal setup

# Go to Modal Dashboard -> Secrets
# Create a Custom secret named "huggingface-secret"
# Add a key named HF_TOKEN and paste your Hugging Face Read Token

# Deploy the backend
modal deploy backend/modal_app.py
```
This will output a live URL for your GPU endpoint.

### 1.2 Alternative: Deploy the CPU Backend (Hugging Face Spaces)
If your Modal credits run out, you can host the model permanently for free on Hugging Face Spaces using `llama.cpp` + Docker SDK on a CPU Basic instance. Note: To preserve CPU latency, visual verification modules (CLIP/OWL-ViT/SymPy) are disabled on this path.

1. Ensure your model files (Base GGUF, mmproj, and LoRA adapter GGUF) are uploaded to a Hugging Face Model repository (e.g., `Shabuuuuuuuuuuu/GSV-Math-GGUF`).
2. Create a new Space on Hugging Face:
   * **SDK:** Docker
   * **Hardware:** CPU Basic (free)
3. Set your Space Secrets in Settings:
   * `HF_TOKEN` = Your Hugging Face read token
   * `API_KEY` = `dev-secret-key` (or matching your frontend api key)
4. Push the contents of the `backend/hf_space/` directory to your Space git repository.
5. Hugging Face will build the container, start `llama-server` in CPU-optimized mode, and launch the FastAPI proxy on port 7860. Your Vercel backend URL will be: `https://<hf-username>-<space-name>.hf.space`.

### 2. Deploy the Frontend (Vercel)
Import the repository into Vercel. During the setup process:
1. Change the **Framework Preset** to `Next.js`
2. Change the **Root Directory** to `frontend`
3. Add an Environment Variable: `NEXT_PUBLIC_MODAL_BACKEND_URL` = `<YOUR_MODAL_URL_FROM_STEP_1>`

### Alternatively: Run Frontend Locally
```bash
cd frontend
npm install

# Set the Modal backend URL
echo "NEXT_PUBLIC_MODAL_BACKEND_URL=https://<YOUR_MODAL_URL>" > .env.local

# Start the dev server
npm run dev
```

## Acknowledgements
* Fine-tuning powered by [Unsloth](https://github.com/unslothai/unsloth)
* VLM architecture provided by [Qwen](https://github.com/QwenLM/Qwen2.5-VL)


