# Security Considerations

## Inference Endpoint
The primary backend endpoint is deployed on Modal at `/solve`. 

**Current Status:** The endpoint is currently exposed and accepts JSON payloads.
**Risks:** Because this endpoint provisions an NVIDIA T4 GPU for several seconds per request, it is vulnerable to cost-exhaustion (denial of wallet) if abused by a script.

### Mitigations
1. **Concurrency Limits:** Modal automatically caps concurrent containers, but we should configure an explicit `concurrency_limit` in `modal_app.py`.
2. **Payload Caps:** The FastAPI endpoint should cap the size of the incoming `image_base64` payload to prevent memory exhaustion before decoding.
3. **Authentication (Planned):** An `X-API-Key` header check using a Modal Secret should be implemented to prevent unauthenticated access.

## External Fetching (SSRF Protection)
The `/solve` endpoint previously supported fetching images via `image_url`. This logic has been restricted/sandboxed to prevent SSRF (Server-Side Request Forgery) attacks where a malicious client instructs the server to probe internal network addresses.

* Fetching is restricted to `http://` and `https://` schemes.
* A strict 10-second timeout is enforced.
* (Recommended) Rely exclusively on `image_base64` payloads passed directly from the frontend to eliminate the fetching surface area entirely.

## CORS
The Modal backend uses `CORSMiddleware` to explicitly allow traffic from the Vercel frontend domain (`gsv-math.vercel.app`) to prevent cross-origin abuse.
