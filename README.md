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
| Fine-Tuned (Tested on Math360k) | 94.00% |

## Live Demo
The application features a custom, lightweight "paper worksheet" UI that interacts directly with the serverless GPU backend.

- **Frontend:** [gsv-math.vercel.app](https://gsv-math-git-main-shabnam311s-projects.vercel.app) 
- **Backend:** Hosted serverlessly on Modal.

## Project Structure
```text
GSV_Math/
├── backend/            # Modal serverless GPU backend (FastAPI)
│   ├── modal_app.py    # Endpoint definitions and Model loading
│   └── pipeline/       # CISC voting, prompt formatting, model inference
├── frontend/           # Next.js React web application
│   ├── src/app/        # Page routing, React components, and CSS
│   └── package.json
└── project_notebooks/  # Training, fine-tuning, and evaluation scripts (Unsloth)
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

