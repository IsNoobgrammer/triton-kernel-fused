# olm-bench
- For noise-redundant runs, use the SEEDS8 set: 23, 24, 12, 2, 9, 28, 69, 2026. Confidence: 0.85
- Set NS dtype per-architecture as a single config value: sm75=fp16, sm120=bf16. Confidence: 0.85
- Use bf16 mixed precision (amp) for model training; keep fp32 option for T4. Confidence: 0.80
- Do not re-run deterministic/known configs — only bench what changed. Confidence: 0.85
- Do not push work-in-progress code that is still under discussion. Confidence: 0.80
- For local smoke tests, cap at 2-3 steps only — never run full scripts locally. Confidence: 0.80
- Use the olm-bench repo (not ablate_muon/) for OLM experiments. Confidence: 0.90
- For optimizer comparisons, use SEEDS8 multi-seed runs to beat the noise floor, not single-seed comparisons. Confidence: 0.85
- Prefer const-LR (no decay) for testing normuon/ema behavior, to distinguish late emergence from decay-specific effects. Confidence: 0.75
