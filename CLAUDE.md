# Claude Code Instructions

## Running GPU Experiments

This repo uses **Modal** for cloud GPU access. The `modal` CLI won't work due to proxy restrictions. Use the Python API instead:

```bash
# Test GPU access
python run_training.py test

# Run training
python run_training.py train --epochs 200
```

## Important: Proxy Patch

The `modal_proxy_patch.py` must be imported BEFORE `modal`. This is already handled in `run_training.py`. If writing new Modal scripts, always:

```python
import modal_proxy_patch  # MUST be first
import modal
# ... rest of code
```

## File Structure

- `modal_proxy_patch.py` - Patches grpclib for HTTP proxy (auto-applies on import)
- `modal_train.py` - Modal functions that run on cloud GPU (uses `src/` via Mount)
- `run_training.py` - Entry point for running Modal functions
- `src/` - Core training code (mounted into Modal container)
