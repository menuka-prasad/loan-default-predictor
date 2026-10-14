Everytime we run mlflow script it will create `mlruns` folder inside parent folder relative to script. so for example if we run `MLflow.ipynb` the `mlruns` folder will creates inside `notebooks` folder.

If we need to point custom location(project root is better), we need to tell it specifically.

```
# 1. Point to the database one folder up
mlflow.set_tracking_uri("sqlite:///../mlflow.db")

experiment_name = "loan_default_prediction"

# 2. Check if the experiment exists. If it doesn't, create it with the custom artifact location.
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(
        name=experiment_name, 
        artifact_location="file:../mlruns"
    )

# 3. Set the experiment as the active one
mlflow.set_experiment(experiment_name)
```

Also since these experiment details are private we dont version control that `mlruns` folder. so we need to update `.gitignore`


## Special Observation
With our `passthrough` columns(which indicated missingness => `rate_of_interest_missing, Interest_rate_spread_missing, Upfront_charges_missing, dtir1_missing`) training model given **1** for all metrics. 
Initially i thought that was from data leakage and checked both X and y(preprocessed data). but there were no issue.

So complete issue was on that missingness indicators. The reason is they were directly corelated with Target column.We essentially handed the model the answer sheet. A logistic regression only needs to look at `rate_of_interest_missing=1` to predict Status=1 with near-perfect certainty — no other features matter.

This is a real form of data leakage — not the classic "test data leaked into training" kind, but **target leakage**: a feature that encodes the target almost perfectly. In production, when a new loan application comes in, we won't know whether `rate_of_interest` will be missing before we make the prediction. The missingness was a property of how the historical data was collected, not something we'd observe at inference time.

So here's the decision we need to make — two options:

- **Option A:** Drop all four missingness indicators entirely. Clean model, no leakage, but we lose that signal.

- **Option B:** Keep `rate_of_interest_missing` but treat it as what it really is — a proxy for *"this loan defaulted and was processed differently."* Only valid if in production we'd genuinely observe whether `rate_of_interest` was submitted or not at application time.


***Question to consider: in a real loan default prediction scenario, at what point in time are we making the prediction? Is it before the loan is issued (to decide whether to approve it), or after (to flag existing loans at risk)?***

At application time, the customer fills out a form. `rate_of_interest` is something the bank sets after evaluating the application, not something the customer provides upfront. So at prediction time, we wouldn't have `rate_of_interest` at all — it doesn't exist yet. The missingness indicator is essentially encoding "this loan was already processed and defaulted" which is information from the future. Classic target leakage.

**Decision: drop all four missingness indicators.**

so now i went to `preprocessing_v1.ipynb` and commented out passthrough cols from ColumnTransformer. We dont need it because it does leaks target.

But now we have real Domain Specific Problem.


We were also keeping `rate_of_interest`, `Interest_rate_spread`, and `Upfront_charges` as features in numeric_cols. 
**Same issue** — these are bank-set values that don't exist at application time.

**So the real question is**: which columns in our dataset are known at application time (customer-provided or verifiable before approval) vs. set after approval?

1. `approv_in_adv` — this means “pre-approved in advance.” At application time, does the bank know this? Actually yes — the customer either came in with a pre-approval or they didn’t. This is known upfront. Keep it.

2. `Neg_ammortization` — this is a loan structure type (does the loan allow negative amortization where unpaid interest gets added to principal). This is a loan product feature the bank decides when structuring the loan, not something known from the application. Remove it.

3. `rate_of_interest`, `Interest_rate_spread`, `Upfront_charges` — all bank-set after evaluation. Remove all three.

4. `loan_limit` — bank-set based on customer profile. This is determined during the approval process, not submitted at application time. Remove it.

5. `interest_only` — bank decision. Remove it.

6. `lump_sum_payment` — you said “predefined by bank when issuing application” — that means it’s set at issuance, which is after the default prediction happens. Remove it.

7. `term` — you said bank sets it based on customer request. The customer requests a term, the bank approves one. At application time you’d have the requested term as input, so this is borderline. For simplicity, keep it — it’s reasonable to assume the requested term is on the application form.

8. `LTV` — both loan_amount and property_value are on the application. Keep it.


## Final column removal list

Remove these from your pipeline entirely — they’re post-approval or bank-set:

1. `rate_of_interest`
2. `Interest_rate_spread`
3. `Upfront_charges`
4. `Neg_ammortization`
5. `loan_limit`
6. `interest_only`
7. `lump_sum_payment`
8. `All four missingness indicator columns`

this is called feature leakage audit and it’s a real step professional ML engineers do before training.

Now we Are going to write the new preprocessing pipeline. (v2)