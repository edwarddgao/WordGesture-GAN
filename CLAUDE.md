# Claude Code Instructions

## Cloud GPU Access

This repo has Modal configured for cloud GPU access. The `modal` CLI won't work due to proxy restrictions. Use the Python API instead.

### Proxy Patch

The `modal_proxy_patch.py` must be imported BEFORE `modal`. Example:

```python
import modal_proxy_patch  # MUST be first - patches grpclib for HTTP proxy
import modal

app = modal.App("my-app")

@app.function(gpu="T4")
def train():
    import torch
    # ... training code
```

### Running Functions

```python
async with app.run():
    result = await train.remote.aio()
```

### Notes

- Modal credentials are set via `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` env vars
- The proxy patch routes gRPC through the HTTP CONNECT proxy
- Use `modal.Mount` to include local code in the container
