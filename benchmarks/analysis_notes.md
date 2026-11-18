# Compute-time analysis

Run `python break_even.py` from the project folder. It writes:

- `benchmarks/break_even.csv`: retrieval prompting (B) versus LoRA (C).
- `benchmarks/accuracy_speed.csv`: largest-budget accuracy and serving speed, including both encoder baselines.

## Break-even definition

Both systems use the same largest label budget, seed 0 for timing, T4 model, batch size 8, software versions, and fixed dev sample. Accuracy uses the first 500 test messages and averages all available seeds; timing uses a separate 200-message dev sample.

For N requests:

```
B allocated serving seconds = N / B throughput
C allocated compute seconds = C training seconds + N / C throughput
N at intersection = C training seconds / (1 / B throughput - 1 / C throughput)
```

The CSV rounds the intersection up to a whole request. If C has no serving-time advantage, there is no finite intersection.

This is an estimated compute-time amortization point at sustained measured throughput, not an observed financial break-even. Serving seconds include the full measured prediction pipeline: B includes CPU query embedding, retrieval, and generation. They represent elapsed allocated-machine time, not GPU kernel time. C training time comes from the saved training runtime.

The calculation assumes compute allocation scales with requests, the measured batch throughput is sustained, and the training and serving machines have the same hourly price if converting time to money. It excludes idle-server bills, model loading, data preparation, labeling, retrieval-index construction, tuning, failed runs, and engineering. It represents training one final adapter, not the cost of the full research study. No dollar savings are claimed.

Different training and serving utilization, low request volume, queueing, or always-on hosting can change the result. Timings are single-session measurements with no confidence interval for the intersection. Keep its integer value as a reproducible calculation, not a precise deployment forecast.

## Accuracy and baseline context

Accuracy and speed must be presented together. Mean accuracy is descriptive; it does not establish statistical significance. The seed counts are recorded explicitly. Invalid or missing category predictions count as incorrect.

DeBERTa is a competing T4 deployment option and must remain visible. BGE plus logistic regression was measured on a Colab CPU; it is a separate hardware option, not a same-device comparison. The recreated DeBERTa checkpoints use the frozen recipe; their timing is paired with the original multi-seed accuracy results rather than a new accuracy evaluation of those checkpoints.

Few-shot uses batch size 1 and is excluded from the primary break-even comparison. Its throughput difference combines prompt length and batching. Zero-shot and few-shot remain in the original benchmark summary.

Prefill/decode profiling was omitted by project decision. End-to-end timing cannot attribute the entire difference to any one generation stage.
