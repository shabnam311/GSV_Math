#!/bin/bash
set -e

echo "=== GSV-Math CPU Backend Starting ==="

# Download GGUF model files from HuggingFace (if not already cached)
echo "Downloading model files..."
python3 -c "
from huggingface_hub import hf_hub_download
import os

REPO = os.environ.get('GGUF_REPO', 'Shabuuuuuuuuuuu/GSV-Math-GGUF')
TOKEN = os.environ.get('HF_TOKEN', None)

files = [
    'Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf',
    'mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf',
    'gsv-math-lora.gguf'
]

for f in files:
    dest = f'/models/{f}'
    if os.path.exists(dest):
        print(f'  ✅ {f} already cached')
    else:
        print(f'  ⬇️  Downloading {f}...')
        hf_hub_download(repo_id=REPO, filename=f, local_dir='/models', token=TOKEN)
        print(f'  ✅ {f} downloaded')

print('All model files ready.')
"

echo "Starting llama-server on port 8081..."
llama-server \
    --model /models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
    --mmproj /models/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf \
    --lora /models/gsv-math-lora.gguf \
    --threads 2 \
    --ctx-size 4096 \
    --n-predict 256 \
    --port 8081 \
    --host 127.0.0.1 &

LLAMA_PID=$!

# Wait for llama-server to be ready
echo "Waiting for llama-server to load model..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:8081/health | grep -q "ok"; then
        echo "✅ llama-server is ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "❌ llama-server failed to start in 120 seconds"
        exit 1
    fi
    sleep 1
done

echo "Starting FastAPI on port 7860..."
python3 -m uvicorn app:app --host 0.0.0.0 --port 7860
