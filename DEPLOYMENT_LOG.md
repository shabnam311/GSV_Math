# Deployment Log

This log tracks material changes to the deployed model, pipeline, or architecture.

## [v1.0] - 2026-08-24
### Added
- Initial decoupled architecture deployment.
- **Frontend:** Static HTML/JS/CSS ready to deploy to Vercel.
- **Backend:** FastAPI, Qwen2.5-VL-7B-Instruct (4-bit), and OWL-ViT structured for Hugging Face Spaces (ZeroGPU).
- **Pipeline:** Implemented CISC (Confidence-Weighted Self-Consistency) voting with k=3-5.
- VRAM protection (automatic image resizing) and GC cleanup implemented on the inference server.
