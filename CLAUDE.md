# Claude Code Instructions

## Running GPU Experiments

This repo uses **Modal** for cloud GPU access. The `modal` CLI won't work due to proxy restrictions. Use the Python API instead:

```bash
python run_training.py test      # Test GPU access
python run_training.py train     # Run training (mounts dataset/ and src/)
python run_training.py list      # List saved checkpoints
```

## Important: Proxy Patch

The `modal_proxy_patch.py` must be imported BEFORE `modal`. This is already handled in `run_training.py`. If writing new Modal scripts:

```python
import modal_proxy_patch  # MUST be first
import modal
```

## File Structure

- `run_training.py` - Entry point (handles proxy patch)
- `modal_train.py` - Modal functions (mounts `src/` and `dataset/`)
- `modal_proxy_patch.py` - Patches grpclib for HTTP proxy
- `src/` - Training code (mounted into container)
- `dataset/` - Dataset files (mounted into container)
