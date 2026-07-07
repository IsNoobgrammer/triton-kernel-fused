# Taste

# coding
- Use SwiGLU activation over GELU. Confidence: 0.85
- For commit messages, end with "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" and use no emoji. Confidence: 0.80
- Always discuss before committing — do not auto-commit or push without the user's approval. Confidence: 0.90

# workflow
- Use the venv from ../BiBo for this repo. Confidence: 0.80
- When benchmarking, ensure all kernels (including compiled baselines) get proper warmup runs, not just fused kernels. Confidence: 0.85
- Include bias update checks in router parity tests. Confidence: 0.85

# documentation
- Keep README and docs professional — never reference .autoresearch or other internal/misc files. Confidence: 0.90
- Document kernel findings informally in .autoresearch directory. Confidence: 0.75

# architecture
- Use sm_XX subdirectories (e.g., kernels/sm75/, kernels/sm120/) for architecture-specific kernels. Confidence: 0.80
- Only keep grouped MoE kernel in sm120; sm75 should not have a grouped variant. Confidence: 0.65
- Memory-conscious fusion: fused kernels should not use significantly more peak memory than the baseline (memory regression is unacceptable). Confidence: 0.90

# olm-bench
See [olm-bench/taste.md](olm-bench/taste.md)
