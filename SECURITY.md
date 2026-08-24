# Security Policy

## Educational & Research Demo
This project and its live demo are provided for educational and research purposes. 

## Data Privacy
- The backend inference service (deployed on Hugging Face Spaces) processes uploaded images in-memory to generate mathematical reasoning traces.
- Uploaded images are **not** persisted to disk or stored beyond the lifecycle of the immediate `/predict` request.
- We do not use user queries or uploaded images to train our models.

## Reporting a Vulnerability
If you discover any security-related issues, please open an issue in the GitHub repository rather than exploiting it. We will address it as quickly as possible.
