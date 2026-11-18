# Statistical analysis

Run `python analyze_results.py`. The script only reads the frozen prediction files and saved splits; it does not train, generate new predictions, or tune settings on test results.

- `metrics.csv`: accuracy, bootstrap interval, descriptive seed-to-seed standard deviation, Emotion macro-F1, and valid-category JSON rate for each configuration.
- `paired_comparisons.csv`: each fine-tuned model versus the best available prompting mean at each label budget.
- `crossovers.csv`: the first tested budget that satisfies the roadmap's fixed rule, separately for LoRA and DeBERTa.

## Fixed crossover rule

The lower end of the fine-tuned model's accuracy interval must exceed the best prompting mean at that budget. Zero-shot is available at every budget. Few-shot is considered only at its measured budget, and B is considered at the matching budget. There is no interpolation between budgets.

The paired-bootstrap difference is a separate supporting measurement. A positive paired lower bound is not substituted for the fixed crossover rule. Neither check requires an arbitrary three-percentage-point gap.

## Confidence interval scope

Both calculations reuse `evaluate.py`: 2,000 bootstrap draws, random seed 0, percentile 95% intervals. The same sampled test-item indices are used for all runs and both systems in a paired comparison. This preserves pairing over messages, even though the systems have different numbers of training seeds.

Seeds are held fixed; the intervals quantify test-item sampling uncertainty conditional on the observed runs. They do not capture uncertainty from drawing new training subsets or from tuning. Seed standard deviation is descriptive and is not a confidence interval.

There are 500 fixed headline test messages per dataset. B has two seeds, fine-tuned models and logistic regression have three, zero-shot has one, and Emotion few-shot at 30/class has only seed 0 because seed 1 ran out of T4 memory. Missing runs are not imputed. Invalid outputs count as incorrect; valid-category JSON requires both the expected structure and a known label.

Intervals are pointwise, not adjusted for multiple budgets or models. Selecting the best prompting mean uses the same test set, so the paired interval does not incorporate comparator-selection uncertainty. Treat crossover locations as descriptive evidence under the fixed rule, not a family-wise 95% guarantee or proof of superiority on other datasets. The script reports the paired interval even when it disagrees with the fixed rule.

Emotion macro-F1 gives equal weight to each of the six true classes. It is averaged over the available seeds. Do not infer rare-class reliability from overall accuracy alone.
