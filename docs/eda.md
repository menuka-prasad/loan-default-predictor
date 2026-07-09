# About Target Variable *Status*

`Status`: *0 = 112,031 (75.3%), 1 = 36,639 (24.7%)* — roughly **3:1** imbalance.
so we cant use accuracy as a metric here. because if model predicts all 0, still it gives around 75% Accuracy.

so we have to pick metric that matches for this dataset

# About Features

### Drop
- ID

## Categorical features

### 1. Ordinal
- age → ['<25','25-34','35-44','45-54','55-64','65-74','>74']
- total_units → ['1U','2U','3U','4U']

### 2. Binary (map to 0/1)
loan_limit, approv_in_adv, Credit_Worthiness, open_credit, business_or_commercial, Neg_ammortization, interest_only, lump_sum_payment, construction_type, Secured_by, co-applicant_credit_type, submission_of_application, Security_Type

### 3. Nominal (one-hot)
Gender, loan_type, loan_purpose, occupancy_type, credit_type, Region

## Numerical Features
**Numeric (already int/float, goes through StandardScaler or similar)**

year, loan_amount, rate_of_interest, Interest_rate_spread, Upfront_charges, term, property_value, income, Credit_Score, LTV, dtir1




# Missing Values

#### 1. `rate_of_interest`
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

### 3. 



## EDA is done. Here's the full strategy locked in:

- Drop: `ID`

### Feature engineering
- Ordinal encode: `age, total_units`
- Binary encode (map to 0/1): `13 binary columns`
- One-hot encode: `Gender, loan_type, loan_purpose, occupancy_type, credit_type, Region`
- Numeric (StandardScaler): `year, loan_amount, property_value, income, Credit_Score, LTV, dtir1, term`

### missing values
- **MNAR columns** — median impute + missingness indicator:  `rate_of_interest, Interest_rate_spread, Upfront_charges`
- Fix before encoding: `Security_Type` typo (Indriect → Indirect)
- Mode impute: `term` (41 missing rows)