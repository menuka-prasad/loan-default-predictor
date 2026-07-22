# About Target Variable *Status*

`Status`: *0 = 112,031 (75.3%), 1 = 36,639 (24.7%)* — roughly **3:1** imbalance.
so we cant use accuracy as a metric here. because if model predicts all 0, still it gives around 75% Accuracy.

so we have to pick metric that matches for this dataset

# About Features

### Drop
- ID

## Categorical features

### 1. Ordinal
- `age` → ['<25','25-34','35-44','45-54','55-64','65-74','>74']
- `total_units` → ['1U','2U','3U','4U']

### 2. Binary (map to 0/1)
`loan_limit, approv_in_adv, Credit_Worthiness, open_credit, business_or_commercial, Neg_ammortization, interest_only, lump_sum_payment, construction_type, Secured_by, co-applicant_credit_type, submission_of_application, Security_Type`

### 3. Nominal (one-hot)
`Gender, loan_type, loan_purpose, occupancy_type, credit_type, Region`

## Numerical Features
**Numeric (already int/float, goes through StandardScaler or similar)**

`year, loan_amount, rate_of_interest, Interest_rate_spread, Upfront_charges, term, property_value, income, Credit_Score, LTV, dtir1`




# Missing Values

```
ID                               0
year                             0
loan_limit                    3344
Gender                           0
approv_in_adv                  908
loan_type                        0
loan_purpose                   134
Credit_Worthiness                0
open_credit                      0
business_or_commercial           0
loan_amount                      0
rate_of_interest             36439
Interest_rate_spread         36639
Upfront_charges              39642
term                            41
Neg_ammortization              121
interest_only                    0
lump_sum_payment                 0
property_value               15098
construction_type                0
occupancy_type                   0
Secured_by                       0
total_units                      0
income                        9150
credit_type                      0
Credit_Score                     0
co-applicant_credit_type         0
age                            200
submission_of_application      200
LTV                          15098
Region                           0
Security_Type                    0
Status                           0
dtir1                        24121
dtype: int64
```

#### 1. `rate_of_interest`, `Interest_rate_spread`, `Upfront_charges`
here missing value count exactly equals to `Status=1` Count.

```
Status
0    112031
1     36639
Name: count, dtype: int64
```

```
rate_of_interest             36439
Interest_rate_spread         36639
Upfront_charges              39642
```

and when look into dataset, we can clearly see all missing values are related to `Status=1` rows. that means there is clear relationship between target variable and this feature.

also this is almost same for `Interest_rate_spread, Upfront_charges` features also. 

**What we can do for this columns**
- Drop these rows - We cant drop that rows since it will drop `Status=1` rows entirely. 
- Drop that columns -  lose potentially important signal because these are high-signal domain features that directly related to loans.
- We can use *Iterative imputation* but it is sophisticated, but risks encoding the MNAR(**Missing Not At Random**) bias into our features. (the model learns "defaulted loans have rate_of_interest = median" which is factually wrong. )
- Keep the columns but add a *missingness indicator* — create a new binary column `rate_of_interest_missing` (**1 if missing, 0 if not**), then impute with median. The model can learn both "what the rate was" AND "whether it was missing at all" as separate signals (this gives the model both pieces of truth: the best available value and an explicit flag that says "this was missing" — so the model can learn that missingness itself is a signal for default risk.)



#### 2. There are same missing counts for property_value (15,098) and LTV (15,098). that is because, *LTV is simply loan_amount/property value.* so if no property value then there is no LTV value.

 so when property_value is missing, LTV can't be computed and is also missing. That's not two separate data quality problems — it's one missing input cascading into one missing output. This matters because it tells you: if you ever impute property_value, you could recompute LTV from loan_amount / property_value instead of separately imputing LTV — more consistent than treating them as independent.

so here we use median imputation to find `property_value` and calculate `LTV` using that.


#### 3. `loan_purpose`, `submission_of_application`, `term`
Only has very few missing values. so mode imputation is better


#### 4. `loan_limit`, `approv_in_adv`, `Neg_ammortization`, `income`, `age`

```python
suspect_cols = ['loan_limit', 'approv_in_adv', 'Neg_ammortization', 'income', 'dtir1', 'age']
for col in suspect_cols:
    result = df.groupby('Status')[col].apply(lambda x: x.isna().mean())
    print(f"\n{col}:\n{result}")

```

```powershell
loan_limit:
Status
0    0.021985
1    0.024045
Name: loan_limit, dtype: float64

approv_in_adv:
Status
0    0.005954
1    0.006578
Name: approv_in_adv, dtype: float64

Neg_ammortization:
Status
0    0.000794
1    0.000873
Name: Neg_ammortization, dtype: float64

income:
Status
0    0.070614
1    0.033816
Name: income, dtype: float64

dtir1:
Status
0    0.069722
1    0.445154
Name: dtir1, dtype: float64

age:
Status
0    0.000000
1    0.005459
Name: age, dtype: float64
```

`loan_limit` — missingness is ~2% for both classes, basically equal. Pure MAR (Missing At Random). Mode impute.

`approv_in_adv` — 0.6% missing, no MNAR pattern. Mode impute.

`Neg_ammortization` — 0.08% missing, no pattern. Mode impute. No reason to drop 121 rows.

`income` — actually less missing for Status=1 (3.8% vs 7%). Slight reverse pattern but not dramatic. Median impute, no indicator needed.

`age` — 0% missing for Status=0, 0.5% for Status=1. Tiny numbers, weak pattern. Mode impute (it's ordinal with a clear most-common bucket).


`dtir1` — 7% missing for non-defaults, 44.5% missing for defaults. That's a strong MNAR signal, same category as `rate_of_interest`. What will be the strategy should be here, given what we already decided for the interest rate columns?

so here we should use median impute + missingness indicator. so we add `dtir1_missing` as a binary indicator column alongside median imputation.

## EDA is done. Here's the full strategy locked in:

- Drop: `ID`
- Fix before encoding: `Security_Type` typo (Indriect → Indirect)

### Feature engineering
- Ordinal encode: `age, total_units`
- Binary encode (map to 0/1): `13 binary columns`
- One-hot encode: `Gender, loan_type, loan_purpose, occupancy_type, credit_type, Region`
- Numeric (StandardScaler): `year, loan_amount, property_value, income, Credit_Score, LTV, dtir1, term`

### missing values
- **MNAR columns** — median impute + missingness indicator:  `rate_of_interest, Interest_rate_spread, Upfront_charges, dtir1`

- Mode impute: `term, loan_limit, approv_in_adv, Neg_ammortization, loan_purpose, submission_of_application, age`
- Median impute: `income`
- Derive after imputing
   - `property_value` → median impute, then recompute `LTV = loan_amount / property_value`
  