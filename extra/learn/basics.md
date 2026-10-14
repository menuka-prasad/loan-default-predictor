## Importance of pipeline

Without a Pipeline, here's concretely what's missing: your StandardScaler instance and your OneHotEncoder/ColumnTransformer instance were fit on training data — they learned the training set's mean/std (for scaling) and which categories exist (for encoding). That fitted state lives in those objects, not in the model. If you only joblib.dump() the model, you've thrown away that learned state entirely.
So at inference time, three different failure modes are possible depending on what you do:

1. You forget to scale/encode at all → raw numbers go straight into XGBoost. It won't crash (XGBoost is tree-based, it'll happily take any numbers) — but predictions will be silently wrong because the model learned on a different numeric scale and encoding.
2. You re-fit a new scaler/encoder on just the one new row → catastrophic. StandardScaler on a single row gives you mean=that value, std=0. Division by zero or garbage output.
3. You hardcode the categories from memory in your API code → fragile, easy to drift from what the model actually saw, and if a new category like "freelance" shows up, it breaks silently or errors unpredictably.

*The fix Pipeline gives you*: it bundles the fitted scaler + encoder + model into one object. joblib.dump(pipeline) saves all the learned state together. At inference, pipeline.predict(raw_row) re-applies the exact same fitted transforms, then predicts — one call, zero chance of mismatch.


## Why does the classification threshold live outside the pipeline/model object instead of inside it? What's different about a threshold compared to a scaler or encoder?

Think about what a scaler/encoder is versus what a threshold is:

- The scaler/encoder were learned from data — fit during training, fixed once trained. They're part of the model's transformation logic.
- The threshold is not learned from data at all. XGBoost outputs a probability (e.g. 0.42 chance of default). The threshold is a business decision you bolt on afterward: "if probability > 0.5, call it a default" — or 0.3, or 0.7, depending on whether false negatives or false positives cost you more.

**Here's the sharper question:** if your model and threshold are bundled into one Pipeline object and you joblib.dump() it together — what would you have to do to change the threshold from 0.5 to 0.35 tomorrow? Would you need to retrain anything?
And the reverse — by keeping it in a separate threshold.json, what can you change without touching the model file at all?

```the threshold is a deployment-time decision, not a training-time artifact. You never need to retrain to change it; you just need to change a number in a file your serving code reads. That's the same category of thinking behind feature flags and config-driven systems in regular software engineering```
