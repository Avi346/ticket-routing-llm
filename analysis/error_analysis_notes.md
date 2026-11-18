# Error analysis

Run `python error_analysis.py` to regenerate the CSV tables. All results use the same first 500 headline test messages, largest label budget, and frozen prediction files. No model was retrained or tuned during this analysis.

## Files and methods

- `per_class.csv`: precision, recall, and F1 computed separately for each run, then averaged over seeds. Support is the number of unique test messages in that class, not multiplied by the number of seeds.
- `worst_classes.csv`: the ten lowest-F1 classes per system and dataset, or all six for Emotion. These are descriptive rankings.
- `top_confusions.csv`: the ten most frequent wrong true-label/predicted-label pairs per system. Counts across runs reuse the same messages; mean errors per seed is included to avoid treating those runs as independent test samples.
- `error_overlap.csv`: matched-seed comparisons of B versus C and C versus DeBERTa. Every row contains 500 messages; the four outcome counts sum to 500. The third C seed is retained for C-versus-DeBERTa but has no matching B run.
- `majority_baseline.csv`: always predict the most frequent dev-set label, then score on the headline test set. The label is selected without inspecting test frequencies. Banking77 is nearly balanced, so its dev-set tie makes that baseline arbitrary; Emotion's dev majority is joy and its test accuracy is 34.8%.
- `review_candidates.csv`: a reproducible uniform sample of 25 messages per dataset from the seed-0 union of B and C errors, using random state 0. It is not a sample of all messages or all encoder errors.
- `error_annotations.csv`: qualitative annotations written after reading all 50 sampled messages, their reference labels, and predictions. The two invalid-category examples were also checked against the saved raw JSON.
- `reviewed_errors.csv`: sampled messages joined to the annotations.
- `review_categories.csv`: counts of the assigned primary error types within this small review sample.

## Findings

### Banking77

Across the two matched seeds, C alone is correct on 26 and 33 messages, while B alone is correct on 20 and 12. Both are wrong on 26 and 24 messages. C improves on B overall, but it does not fix every B error and introduces some of its own.

Repeated confusions involve related intent boundaries: card arrival versus delivery estimate, currency support versus exchanging money, and transfer failure versus top-up processing states. C's card-arrival F1 averages about 0.50, but this class has only six headline test messages. Do not generalize a class ranking from such small support.

Some errors are exact-category failures. In the review sample C generated `reverted_transfer` and B generated `top_up_by_card`, neither of which is an allowed category. Both replies were structurally valid JSON. Therefore valid JSON syntax alone would overstate usable-output validity.

### Emotion

Across the matched seeds, C alone is correct on 127 and 124 messages, versus 17 and 14 for B alone. Both are wrong on 33 and 31. The improvement is much broader than on Banking77.

B frequently confuses joy with sadness or love. For example, joy-to-sadness errors average 22 messages per seed. Fine-tuning reduces many of these errors, but both C and DeBERTa still confuse joy with love and fear with surprise.

Mean macro-F1 at this budget is approximately 0.636 for B, 0.874 for C, 0.877 for DeBERTa, and 0.496 for LogReg. This supports the accuracy finding while showing remaining weaknesses on minority classes. Surprise has only 17 test messages; its F1 is about 0.729 for C and 0.736 for DeBERTa, below their overall accuracy.

The qualitative sample contains mixed emotions, negation, vague wording, and boundaries such as love versus joy. Some predictions are reasonable alternative readings of the text even when they disagree with the reference. These observations are consistent with dataset-specific label conventions being important, but do not prove the mechanism by which fine-tuning helps.

C and DeBERTa also make different mistakes: C alone is correct on 15-21 messages per matched run, while DeBERTa alone is correct on 14-25. Their similar overall Emotion accuracy does not mean identical predictions.

## Limits of the qualitative review

Annotations are AI-assisted exploratory judgments made with the reference labels visible, not a blinded human annotation study. Categories describe the most salient issue and can overlap conceptually. They are not validated causal explanations or estimates of dataset label noise. No reference label was changed. Review-category frequencies apply only to the sampled union of B/C errors; do not present them as population error rates.

Classwise results and confusion pairs are descriptive, without significance claims. The headline crossover conclusions remain those in `crossovers.csv`.
