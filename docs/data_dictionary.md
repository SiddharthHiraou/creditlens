# Data Dictionary

**Generated** by `make docs` from the feature spec, the information-value
table and the global SHAP summary. Do not edit by hand — a hand-maintained
feature list disagrees with the model within a sprint.

- Feature spec version **1**, fingerprint `19cc7282140b4dd2`
- **221** features built, **72** selected into the champion
- **47** carry a monotonic constraint
- **212** mapped to an ECOA reason family, **9** suppressed from disclosure

## Reading the columns

| Column | Meaning |
|---|---|
| **In model** | Survived selection into the champion's feature spec |
| **IV** | Information value on training data. Below 0.02 is unpredictive |
| **Direction** | Monotonic constraint. `↑ risk` means PD may only rise with the feature |
| **SHAP share** | Share of total mean-absolute SHAP across the out-of-time fold |
| **Reason family** | ECOA family this feature is disclosed under, or `suppressed` |
| **Null %** | Share missing across the full population |

Missingness is meaningful throughout: a null bureau aggregate means the
applicant has no bureau file, which is a real and risk-relevant segment,
not a data quality problem. History *counts* fill to zero; *ratios* and
*slopes* stay null.

## `POS_CASH_balance`

Point-of-sale and cash loan servicing. **15 features, 4 in the champion.**

| Feature | In model | IV | Direction | SHAP share | Reason family | Null % |
|---|---|---|---|---|---|---|
| `POS_cnt_future_latest` | no | 0.0055 (unpredictive) | unconstrained | — | credit_experience | 20.1% |
| `POS_cnt_future_mean` | no | 0.0081 (unpredictive) | unconstrained | — | debt_burden | 20.1% |
| `POS_cnt_instalment_max` | no | 0.0036 (unpredictive) | unconstrained | — | debt_burden | 20.1% |
| `POS_cnt_instalment_mean` | no | 0.0067 (unpredictive) | unconstrained | — | debt_burden | 20.1% |
| `POS_completed_share` | no | 0.0022 (unpredictive) | unconstrained | — | credit_experience | 20.1% |
| `POS_days_since_last` | no | 0.0013 (unpredictive) | unconstrained | — | credit_experience | 20.1% |
| `POS_dpd_max` | yes | 0.1345 (medium) | ↑ risk | 1.07% | repayment_history | 20.1% |
| `POS_dpd_mean` | yes | 0.1851 (medium) | ↑ risk | 0.98% | repayment_history | 20.1% |
| `POS_dpd_share` | yes | 0.2077 (medium) | ↑ risk | 1.62% | repayment_history | 20.1% |
| `POS_dpd_slope` | no | 0.1231 (medium) | unconstrained | — | repayment_history | 20.1% |
| `POS_n_active_months` | no | 0.0044 (unpredictive) | unconstrained | — | credit_experience | 0.0% |
| `POS_n_dpd_months` | yes | 0.1556 (medium) | ↑ risk | 0.23% | repayment_history | 0.0% |
| `POS_n_dpd_over_30` | no | 0.1470 (medium) | ↑ risk | — | repayment_history | 0.0% |
| `POS_n_loans` | no | 0.0026 (unpredictive) | unconstrained | — | credit_experience | 0.0% |
| `POS_n_months` | no | 0.0052 (unpredictive) | unconstrained | — | credit_experience | 0.0% |

## `application`

Amounts stated on the application. **31 features, 8 in the champion.**

| Feature | In model | IV | Direction | SHAP share | Reason family | Null % |
|---|---|---|---|---|---|---|
| `AMT_ANNUITY` | no | 0.1403 (medium) | unconstrained | — | loan_structure | 0.0% |
| `AMT_CREDIT` | no | 0.1275 (medium) | unconstrained | — | loan_structure | 0.0% |
| `AMT_GOODS_PRICE` | no | 0.1233 (medium) | unconstrained | — | loan_structure | 0.0% |
| `AMT_INCOME_TOTAL` | yes | 0.2113 (medium) | unconstrained | 1.81% | affordability | 0.0% |
| `CNT_CHILDREN` | no | 0.0001 (unpredictive) | unconstrained | — | **suppressed** | 0.0% |
| `CNT_FAM_MEMBERS` | no | 0.0005 (unpredictive) | unconstrained | — | **suppressed** | 0.0% |
| `CODE_GENDER` | no | 0.0002 (unpredictive) | unconstrained | — | **suppressed** | 0.0% |
| `DAYS_BIRTH` | yes | 0.1253 (medium) | unconstrained | 2.58% | **suppressed** | 0.0% |
| `DAYS_EMPLOYED` | no | 0.1861 (medium) | unconstrained | — | employment_stability | 0.0% |
| `DAYS_ID_PUBLISH` | no | 0.0016 (unpredictive) | unconstrained | — | **suppressed** | 0.0% |
| `EXT_1_missing` | no | 0.0043 (unpredictive) | unconstrained | — | external_score | 0.0% |
| `EXT_2_missing` | no | 0.0000 (unpredictive) | unconstrained | — | external_score | 0.0% |
| `EXT_3_missing` | no | 0.0162 (unpredictive) | unconstrained | — | external_score | 0.0% |
| `EXT_SOURCE_1` | no | 0.1620 (medium) | ↓ risk | — | external_score | 56.0% |
| `EXT_SOURCE_2` | yes | 0.5418 (suspicious) | ↓ risk | 5.84% | external_score | 3.3% |
| `EXT_SOURCE_3` | no | 0.2208 (medium) | ↓ risk | — | external_score | 20.5% |
| `EXT_all_missing` | no | 0.0000 (unpredictive) | ↑ risk | — | external_score | 0.0% |
| `EXT_max` | yes | 0.5014 (suspicious) | ↓ risk | 2.25% | external_score | 0.5% |
| `EXT_mean` | yes | 0.5747 (suspicious) | ↓ risk | 3.40% | external_score | 0.5% |
| `EXT_mean_x_age` | yes | 0.6199 (suspicious) | unconstrained | 9.09% | external_score | 0.5% |
| `EXT_mean_x_annuity_ratio` | yes | 0.0785 (weak) | unconstrained | 1.47% | external_score | 0.5% |
| `EXT_min` | no | 0.4839 (strong) | ↓ risk | — | external_score | 0.5% |
| `EXT_n_present` | no | 0.0347 (weak) | ↓ risk | — | external_score | 0.0% |
| `EXT_spread` | no | 0.0292 (weak) | unconstrained | — | external_score | 0.5% |
| `FLAG_OWN_CAR` | no | 0.0009 (unpredictive) | unconstrained | — | debt_burden | 0.0% |
| `FLAG_OWN_REALTY` | no | 0.0000 (unpredictive) | unconstrained | — | debt_burden | 0.0% |
| `NAME_CONTRACT_TYPE` | no | 0.0007 (unpredictive) | unconstrained | — | loan_structure | 0.0% |
| `NAME_EDUCATION_TYPE` | no | 0.1251 (medium) | unconstrained | — | employment_stability | 0.0% |
| `NAME_FAMILY_STATUS` | no | 0.0003 (unpredictive) | unconstrained | — | **suppressed** | 0.0% |
| `OCCUPATION_TYPE` | no | 0.0020 (unpredictive) | unconstrained | — | employment_stability | 0.0% |
| `REGION_RATING_CLIENT` | yes | 0.1849 (medium) | unconstrained | 5.44% | residency_region | 0.0% |

## `bureau`

Prior credit lines reported by the credit bureau. **38 features, 11 in the champion.**

| Feature | In model | IV | Direction | SHAP share | Reason family | Null % |
|---|---|---|---|---|---|---|
| `BURO_active_credit_total` | no | 0.0759 (weak) | unconstrained | — | debt_burden | 3.1% |
| `BURO_active_debt_to_credit` | no | 0.0547 (weak) | ↑ risk | — | debt_burden | 20.2% |
| `BURO_active_debt_total` | no | 0.0703 (weak) | unconstrained | — | debt_burden | 3.1% |
| `BURO_active_overdue_total` | no | 0.0214 (weak) | ↑ risk | — | bureau_delinquency | 3.1% |
| `BURO_active_share` | yes | 0.0641 (weak) | unconstrained | 0.34% | credit_experience | 3.1% |
| `BURO_amt_sum_max` | no | 0.0658 (weak) | unconstrained | — | debt_burden | 3.1% |
| `BURO_amt_sum_mean` | yes | 0.0468 (weak) | unconstrained | 0.66% | debt_burden | 3.1% |
| `BURO_amt_sum_total` | no | 0.1143 (medium) | unconstrained | — | debt_burden | 3.1% |
| `BURO_days_credit_mean` | yes | 0.0464 (weak) | unconstrained | 0.57% | credit_experience | 3.1% |
| `BURO_days_credit_std` | no | 0.0862 (weak) | unconstrained | — | credit_experience | 12.2% |
| `BURO_days_overdue_max` | no | 0.1411 (medium) | ↑ risk | — | bureau_delinquency | 3.1% |
| `BURO_days_overdue_mean` | yes | 0.1409 (medium) | ↑ risk | 0.49% | bureau_delinquency | 3.1% |
| `BURO_days_since_first_line` | no | 0.0671 (weak) | unconstrained | — | credit_experience | 3.1% |
| `BURO_days_since_last_line` | no | 0.0631 (weak) | unconstrained | — | credit_experience | 3.1% |
| `BURO_debt_max` | no | 0.0628 (weak) | unconstrained | — | debt_burden | 3.1% |
| `BURO_debt_mean` | no | 0.0546 (weak) | unconstrained | — | debt_burden | 3.1% |
| `BURO_debt_recency_wtd` | no | 0.0688 (weak) | unconstrained | — | debt_burden | 3.1% |
| `BURO_debt_recency_wtd_norm` | no | 0.0532 (weak) | unconstrained | — | debt_burden | 3.1% |
| `BURO_debt_to_credit` | no | 0.0561 (weak) | ↑ risk | — | debt_burden | 3.1% |
| `BURO_debt_total` | no | 0.0703 (weak) | unconstrained | — | debt_burden | 3.1% |
| `BURO_lines_per_year` | no | 0.1230 (medium) | unconstrained | — | credit_experience | 3.1% |
| `BURO_n_active` | no | 0.0880 (weak) | unconstrained | — | credit_experience | 0.0% |
| `BURO_n_closed` | yes | 0.1068 (medium) | unconstrained | 0.24% | credit_experience | 0.0% |
| `BURO_n_credit_types` | no | 0.0902 (weak) | unconstrained | — | credit_experience | 0.0% |
| `BURO_n_lines` | no | 0.1816 (medium) | unconstrained | — | credit_experience | 0.0% |
| `BURO_n_overdue_lines` | yes | 0.1328 (medium) | ↑ risk | 0.32% | bureau_delinquency | 0.0% |
| `BURO_n_type_car_loan` | no | 0.0137 (unpredictive) | unconstrained | — | credit_experience | 0.0% |
| `BURO_n_type_consumer_credit` | no | 0.1019 (medium) | unconstrained | — | credit_experience | 0.0% |
| `BURO_n_type_credit_card` | no | 0.0541 (weak) | unconstrained | — | credit_experience | 0.0% |
| `BURO_n_type_microloan` | no | 0.0150 (unpredictive) | unconstrained | — | credit_experience | 0.0% |
| `BURO_n_type_mortgage` | no | 0.0092 (unpredictive) | unconstrained | — | credit_experience | 0.0% |
| `BURO_overdue_line_share` | yes | 0.1402 (medium) | ↑ risk | 0.24% | bureau_delinquency | 3.1% |
| `BURO_overdue_max` | no | 0.1420 (medium) | ↑ risk | — | bureau_delinquency | 3.1% |
| `BURO_overdue_recency_wtd` | yes | 0.1454 (medium) | unconstrained | 0.45% | bureau_delinquency | 3.1% |
| `BURO_overdue_to_debt` | yes | 0.1373 (medium) | ↑ risk | 0.59% | bureau_delinquency | 20.2% |
| `BURO_overdue_total` | yes | 0.1436 (medium) | ↑ risk | 0.49% | bureau_delinquency | 3.1% |
| `BURO_prolong_total` | yes | 0.0371 (weak) | ↑ risk | 0.67% | bureau_delinquency | 3.1% |
| `BURO_recency_weight_total` | no | 0.0802 (weak) | unconstrained | — | credit_experience | 3.1% |

## `bureau_balance`

Monthly bureau status history per credit line. **18 features, 14 in the champion.**

| Feature | In model | IV | Direction | SHAP share | Reason family | Null % |
|---|---|---|---|---|---|---|
| `BB_closed_share` | yes | 0.2362 (medium) | unconstrained | 1.39% | credit_experience | 3.1% |
| `BB_late_share_12m` | yes | 0.3549 (strong) | ↑ risk | 2.07% | bureau_delinquency | 3.1% |
| `BB_late_share_3m` | yes | 0.2044 (medium) | ↑ risk | 0.23% | bureau_delinquency | 3.1% |
| `BB_late_share_6m` | yes | 0.2846 (medium) | ↑ risk | 0.91% | bureau_delinquency | 3.1% |
| `BB_late_share_total` | yes | 0.4355 (strong) | ↑ risk | 3.97% | bureau_delinquency | 3.1% |
| `BB_max_status` | yes | 0.3411 (strong) | ↑ risk | 2.10% | bureau_delinquency | 3.1% |
| `BB_max_status_12m` | no | 0.3260 (strong) | ↑ risk | — | bureau_delinquency | 3.1% |
| `BB_max_status_3m` | yes | 0.2656 (medium) | ↑ risk | 0.92% | bureau_delinquency | 3.1% |
| `BB_max_status_6m` | yes | 0.3066 (strong) | ↑ risk | 0.91% | bureau_delinquency | 3.1% |
| `BB_mean_status` | yes | 0.4103 (strong) | ↑ risk | 1.09% | bureau_delinquency | 3.1% |
| `BB_n_late_12m` | yes | 0.4489 (strong) | ↑ risk | 1.56% | bureau_delinquency | 0.0% |
| `BB_n_late_3m` | yes | 0.2897 (medium) | ↑ risk | 0.51% | bureau_delinquency | 0.0% |
| `BB_n_late_6m` | yes | 0.3710 (strong) | ↑ risk | 0.50% | bureau_delinquency | 0.0% |
| `BB_n_late_total` | yes | 0.4998 (strong) | ↑ risk | 2.07% | bureau_delinquency | 0.0% |
| `BB_n_lines` | no | 0.1816 (medium) | unconstrained | — | credit_experience | 0.0% |
| `BB_n_months` | no | 0.1689 (medium) | unconstrained | — | credit_experience | 0.0% |
| `BB_status_slope` | yes | 0.2293 (medium) | unconstrained | 0.38% | bureau_delinquency | 3.1% |
| `BB_unknown_share` | no | 0.0457 (weak) | unconstrained | — | credit_experience | 3.1% |

## `credit_card_balance`

Revolving account balances and utilisation. **32 features, 6 in the champion.**

| Feature | In model | IV | Direction | SHAP share | Reason family | Null % |
|---|---|---|---|---|---|---|
| `CC_balance_max` | no | 0.0057 (unpredictive) | unconstrained | — | utilisation | 50.1% |
| `CC_balance_mean` | no | 0.0078 (unpredictive) | unconstrained | — | utilisation | 50.1% |
| `CC_balance_slope` | no | 0.1049 (medium) | unconstrained | — | utilisation | 50.1% |
| `CC_dpd_max` | no | 0.0826 (weak) | ↑ risk | — | utilisation | 50.1% |
| `CC_dpd_share` | yes | 0.1187 (medium) | ↑ risk | 0.33% | utilisation | 50.1% |
| `CC_drawings_mean` | no | 0.0039 (unpredictive) | unconstrained | — | utilisation | 50.1% |
| `CC_drawings_total` | no | 0.0060 (unpredictive) | unconstrained | — | utilisation | 50.1% |
| `CC_limit_max` | no | 0.0055 (unpredictive) | unconstrained | — | utilisation | 50.1% |
| `CC_limit_mean` | no | 0.0033 (unpredictive) | unconstrained | — | utilisation | 50.1% |
| `CC_n_cards` | no | 0.0012 (unpredictive) | unconstrained | — | credit_experience | 0.0% |
| `CC_n_dpd_12m` | no | 0.0741 (weak) | unconstrained | — | utilisation | 0.0% |
| `CC_n_dpd_3m` | no | 0.0386 (weak) | unconstrained | — | utilisation | 0.0% |
| `CC_n_dpd_6m` | yes | 0.0573 (weak) | unconstrained | 0.31% | utilisation | 0.0% |
| `CC_n_dpd_months` | yes | 0.0759 (weak) | ↑ risk | 0.26% | utilisation | 0.0% |
| `CC_n_months` | no | 0.0034 (unpredictive) | unconstrained | — | credit_experience | 0.0% |
| `CC_n_months_over_limit` | no | 0.0000 (unpredictive) | ↑ risk | — | utilisation | 0.0% |
| `CC_payment_mean` | no | 0.0061 (unpredictive) | unconstrained | — | utilisation | 50.1% |
| `CC_payment_to_drawings` | no | 0.0029 (unpredictive) | unconstrained | — | utilisation | 50.1% |
| `CC_payment_total` | no | 0.0062 (unpredictive) | unconstrained | — | utilisation | 50.1% |
| `CC_share_months_over_90pct` | no | 0.0027 (unpredictive) | ↑ risk | — | utilisation | 50.1% |
| `CC_util_latest` | yes | 0.0483 (weak) | ↑ risk | 0.11% | utilisation | 50.1% |
| `CC_util_max` | no | 0.0125 (unpredictive) | ↑ risk | — | utilisation | 50.1% |
| `CC_util_max_12m` | no | 0.0538 (weak) | unconstrained | — | utilisation | 50.1% |
| `CC_util_max_3m` | no | 0.0756 (weak) | unconstrained | — | utilisation | 50.1% |
| `CC_util_max_6m` | no | 0.0712 (weak) | unconstrained | — | utilisation | 50.1% |
| `CC_util_mean` | no | 0.0047 (unpredictive) | ↑ risk | — | utilisation | 50.1% |
| `CC_util_mean_12m` | yes | 0.1379 (medium) | unconstrained | 0.55% | utilisation | 50.1% |
| `CC_util_mean_3m` | no | 0.1283 (medium) | unconstrained | — | utilisation | 50.1% |
| `CC_util_mean_6m` | no | 0.1595 (medium) | unconstrained | — | utilisation | 50.1% |
| `CC_util_recent_vs_overall` | yes | 0.1275 (medium) | unconstrained | 0.31% | utilisation | 50.1% |
| `CC_util_slope` | no | 0.1306 (medium) | ↑ risk | — | utilisation | 50.1% |
| `CC_util_std` | no | 0.0085 (unpredictive) | unconstrained | — | utilisation | 50.1% |

## `derived (application)`

Affordability ratios from the application. **19 features, 7 in the champion.**

| Feature | In model | IV | Direction | SHAP share | Reason family | Null % |
|---|---|---|---|---|---|---|
| `RATIO_annuity_to_credit` | no | 0.0337 (weak) | ↑ risk | — | affordability | 0.0% |
| `RATIO_annuity_to_income` | yes | 0.3057 (strong) | ↑ risk | 4.59% | affordability | 0.0% |
| `RATIO_children_to_family` | no | 0.0004 (unpredictive) | unconstrained | — | **suppressed** | 0.0% |
| `RATIO_credit_to_goods` | no | 0.0040 (unpredictive) | unconstrained | — | loan_structure | 0.0% |
| `RATIO_credit_to_income` | yes | 0.3017 (strong) | ↑ risk | 2.32% | affordability | 0.0% |
| `RATIO_downpayment` | no | 0.0040 (unpredictive) | ↓ risk | — | loan_structure | 0.0% |
| `RATIO_goods_to_income` | no | 0.2963 (medium) | ↑ risk | — | affordability | 0.0% |
| `RATIO_implied_term` | no | 0.0337 (weak) | unconstrained | — | loan_structure | 0.0% |
| `RATIO_income_per_family_member` | yes | 0.1322 (medium) | ↓ risk | 1.13% | affordability | 0.0% |
| `RATIO_residual_income` | yes | 0.2569 (medium) | ↓ risk | 1.75% | affordability | 0.0% |
| `RATIO_residual_income_per_member` | yes | 0.2371 (medium) | ↓ risk | 2.19% | affordability | 0.0% |
| `STAB_age_at_employment_start` | yes | 0.0456 (weak) | unconstrained | 0.94% | employment_stability | 0.0% |
| `STAB_age_band` | no | 0.1241 (medium) | unconstrained | — | **suppressed** | 0.0% |
| `STAB_age_years` | no | 0.1252 (medium) | unconstrained | — | **suppressed** | 0.0% |
| `STAB_employed_over_5y` | no | 0.0849 (weak) | ↓ risk | — | employment_stability | 0.0% |
| `STAB_employed_share_of_adult_life` | no | 0.1083 (medium) | ↓ risk | — | employment_stability | 0.0% |
| `STAB_employed_under_1y` | no | 0.0000 (unpredictive) | ↑ risk | — | employment_stability | 0.0% |
| `STAB_employed_years` | yes | 0.1873 (medium) | ↓ risk | 6.09% | employment_stability | 0.0% |
| `STAB_id_published_years` | no | 0.0014 (unpredictive) | unconstrained | — | credit_experience | 0.0% |

## `derived (cross-source)`

Relates the application to existing obligations. **5 features, 1 in the champion.**

| Feature | In model | IV | Direction | SHAP share | Reason family | Null % |
|---|---|---|---|---|---|---|
| `XSRC_annuity_to_prev_annuity` | yes | 0.0934 (weak) | unconstrained | 1.01% | loan_structure | 4.4% |
| `XSRC_bureau_debt_to_income` | no | 0.1388 (medium) | ↑ risk | — | affordability | 3.1% |
| `XSRC_credit_to_prev_credit` | no | 0.1058 (medium) | unconstrained | — | loan_structure | 10.2% |
| `XSRC_new_credit_to_bureau_debt` | no | 0.0517 (weak) | unconstrained | — | loan_structure | 20.2% |
| `XSRC_total_debt_to_income` | no | 0.3246 (strong) | ↑ risk | — | affordability | 3.1% |

## `derived (join)`

Marks an applicant with no history in a source. **5 features, 0 in the champion.**

| Feature | In model | IV | Direction | SHAP share | Reason family | Null % |
|---|---|---|---|---|---|---|
| `FLAG_no_buro_history` | no | 0.0000 (unpredictive) | unconstrained | — | credit_experience | 0.0% |
| `FLAG_no_cc_history` | no | 0.0009 (unpredictive) | unconstrained | — | credit_experience | 0.0% |
| `FLAG_no_inst_history` | no | 0.0000 (unpredictive) | unconstrained | — | credit_experience | 0.0% |
| `FLAG_no_pos_history` | no | 0.0013 (unpredictive) | unconstrained | — | credit_experience | 0.0% |
| `FLAG_no_prev_history` | no | 0.0000 (unpredictive) | unconstrained | — | credit_experience | 0.0% |

## `installments_payments`

Actual repayment conduct on prior loans. **34 features, 15 in the champion.**

| Feature | In model | IV | Direction | SHAP share | Reason family | Null % |
|---|---|---|---|---|---|---|
| `INST_amt_due_mean` | no | 0.0046 (unpredictive) | unconstrained | — | debt_burden | 10.2% |
| `INST_amt_due_total` | no | 0.0059 (unpredictive) | unconstrained | — | debt_burden | 10.2% |
| `INST_amt_paid_total` | no | 0.0039 (unpredictive) | unconstrained | — | debt_burden | 10.2% |
| `INST_days_late_slope` | no | 0.0208 (weak) | unconstrained | — | repayment_history | 10.2% |
| `INST_days_since_first_due` | no | 0.0029 (unpredictive) | unconstrained | — | credit_experience | 10.2% |
| `INST_days_since_last_due` | no | 0.0053 (unpredictive) | unconstrained | — | credit_experience | 10.2% |
| `INST_late_share_12m` | yes | 0.0786 (weak) | ↑ risk | 0.33% | repayment_history | 15.6% |
| `INST_late_share_24m` | yes | 0.1371 (medium) | ↑ risk | 1.35% | repayment_history | 11.2% |
| `INST_late_share_6m` | yes | 0.0518 (weak) | ↑ risk | 0.63% | repayment_history | 27.8% |
| `INST_late_share_total` | yes | 0.2600 (medium) | ↑ risk | 3.27% | repayment_history | 10.2% |
| `INST_max_days_late` | no | 0.0931 (weak) | ↑ risk | — | repayment_history | 10.2% |
| `INST_max_days_late_12m` | yes | 0.0782 (weak) | unconstrained | 1.01% | repayment_history | 15.6% |
| `INST_max_days_late_24m` | yes | 0.0973 (weak) | unconstrained | 0.37% | repayment_history | 11.2% |
| `INST_max_days_late_6m` | yes | 0.0511 (weak) | unconstrained | 0.55% | repayment_history | 27.8% |
| `INST_mean_days_late` | no | 0.2126 (medium) | ↑ risk | — | repayment_history | 10.2% |
| `INST_mean_days_late_when_late` | no | 0.0559 (weak) | unconstrained | — | repayment_history | 21.0% |
| `INST_n_late_12m` | no | 0.0827 (weak) | unconstrained | — | repayment_history | 0.0% |
| `INST_n_late_24m` | yes | 0.1135 (medium) | unconstrained | 0.39% | repayment_history | 0.0% |
| `INST_n_late_6m` | yes | 0.0532 (weak) | unconstrained | 0.12% | repayment_history | 0.0% |
| `INST_n_late_over_30d` | yes | 0.0589 (weak) | ↑ risk | 0.06% | repayment_history | 0.0% |
| `INST_n_late_over_90d` | no | 0.0000 (unpredictive) | ↑ risk | — | repayment_history | 0.0% |
| `INST_n_late_total` | yes | 0.1491 (medium) | ↑ risk | 0.33% | repayment_history | 0.0% |
| `INST_n_loans` | no | 0.0028 (unpredictive) | unconstrained | — | credit_experience | 0.0% |
| `INST_n_payments` | no | 0.0046 (unpredictive) | unconstrained | — | credit_experience | 0.0% |
| `INST_n_short_12m` | no | 0.0550 (weak) | unconstrained | — | repayment_history | 0.0% |
| `INST_n_short_24m` | no | 0.0958 (weak) | unconstrained | — | repayment_history | 0.0% |
| `INST_n_short_6m` | no | 0.0256 (weak) | unconstrained | — | repayment_history | 0.0% |
| `INST_n_short_total` | no | 0.1783 (medium) | ↑ risk | — | repayment_history | 0.0% |
| `INST_paid_to_due` | yes | 0.1955 (medium) | ↓ risk | 1.97% | repayment_history | 10.2% |
| `INST_payments_per_loan` | no | 0.0055 (unpredictive) | unconstrained | — | credit_experience | 10.2% |
| `INST_short_share_total` | yes | 0.2246 (medium) | ↑ risk | 1.47% | repayment_history | 10.2% |
| `INST_shortfall_max` | yes | 0.1446 (medium) | ↑ risk | 1.03% | repayment_history | 10.2% |
| `INST_shortfall_total` | yes | 0.1774 (medium) | ↑ risk | 0.36% | repayment_history | 10.2% |
| `INST_std_days_late` | no | 0.1525 (medium) | unconstrained | — | repayment_history | 10.2% |

## `previous_application`

The applicant's history with this lender. **24 features, 6 in the champion.**

| Feature | In model | IV | Direction | SHAP share | Reason family | Null % |
|---|---|---|---|---|---|---|
| `PREV_amt_annuity_max` | yes | 0.0212 (weak) | unconstrained | 0.52% | prior_applications | 4.4% |
| `PREV_amt_annuity_mean` | no | 0.0191 (unpredictive) | unconstrained | — | prior_applications | 4.4% |
| `PREV_amt_application_max` | no | 0.0207 (weak) | unconstrained | — | prior_applications | 4.4% |
| `PREV_amt_application_mean` | no | 0.0194 (unpredictive) | unconstrained | — | prior_applications | 4.4% |
| `PREV_amt_application_total` | yes | 0.0342 (weak) | unconstrained | 0.79% | prior_applications | 4.4% |
| `PREV_amt_credit_mean` | no | 0.0348 (weak) | unconstrained | — | prior_applications | 4.4% |
| `PREV_amt_credit_total` | no | 0.0112 (unpredictive) | unconstrained | — | prior_applications | 4.4% |
| `PREV_applications_per_year` | no | 0.0333 (weak) | unconstrained | — | prior_applications | 4.4% |
| `PREV_approved_amt_mean` | no | 0.0039 (unpredictive) | unconstrained | — | prior_applications | 10.2% |
| `PREV_cnt_payment_max` | no | 0.0246 (weak) | unconstrained | — | prior_applications | 4.4% |
| `PREV_cnt_payment_mean` | no | 0.0195 (unpredictive) | unconstrained | — | prior_applications | 4.4% |
| `PREV_credit_to_application` | yes | 0.0730 (weak) | unconstrained | 2.11% | prior_applications | 4.4% |
| `PREV_days_decision_mean` | no | 0.0221 (weak) | unconstrained | — | prior_applications | 4.4% |
| `PREV_days_since_first` | no | 0.0209 (weak) | unconstrained | — | prior_applications | 4.4% |
| `PREV_days_since_last` | no | 0.0227 (weak) | unconstrained | — | prior_applications | 4.4% |
| `PREV_last_was_refused` | yes | 0.0426 (weak) | ↑ risk | 0.41% | prior_applications | 4.4% |
| `PREV_n_applications` | no | 0.0546 (weak) | unconstrained | — | prior_applications | 0.0% |
| `PREV_n_approved` | no | 0.0028 (unpredictive) | unconstrained | — | prior_applications | 0.0% |
| `PREV_n_canceled` | no | 0.0044 (unpredictive) | unconstrained | — | prior_applications | 0.0% |
| `PREV_n_contract_types` | no | 0.0331 (weak) | unconstrained | — | prior_applications | 0.0% |
| `PREV_n_refused` | yes | 0.1324 (medium) | ↑ risk | 0.87% | prior_applications | 0.0% |
| `PREV_n_unused` | no | 0.0020 (unpredictive) | unconstrained | — | prior_applications | 0.0% |
| `PREV_refusal_rate` | yes | 0.0952 (weak) | ↑ risk | 0.74% | prior_applications | 4.4% |
| `PREV_refused_amt_mean` | no | 0.1109 (medium) | unconstrained | — | prior_applications | 62.8% |

## Why features were dropped

Selection runs on **training data only**. Screening on the full frame — even
just to compute a correlation matrix — leaks out-of-time information into the
choice of features, which is subtle enough to survive review.

| Reason | Features dropped |
|---|---|
| no lift over shuffled-target baseline | 70 |
| iv<0.02 | 68 |
| corr>0.95 with BURO_debt_total | 2 |
| corr>0.95 with RATIO_credit_to_income | 1 |
| corr>0.95 with STAB_employed_years | 1 |
| corr>0.95 with BURO_n_lines | 1 |
| corr>0.95 with POS_n_dpd_months | 1 |
| corr>0.95 with BURO_overdue_total | 1 |
| corr>0.95 with DAYS_BIRTH | 1 |
| corr>0.95 with AMT_CREDIT | 1 |

## Suppressed from disclosure

These may drive the model but can never appear in an adverse action notice.
Age, sex and family status are protected or proxy-protected under ECOA.

- `CNT_CHILDREN`
- `CNT_FAM_MEMBERS`
- `CODE_GENDER`
- `DAYS_BIRTH`
- `DAYS_ID_PUBLISH`
- `NAME_FAMILY_STATUS`
- `RATIO_children_to_family`
- `STAB_age_band`
- `STAB_age_years`

Note `EXT_mean_x_age`, the model's strongest single feature, is **not** on
this list. It embeds age through an interaction and is disclosed under the
external-score family, so the applicant is told something they can act on and
age is never named as a basis for denial.
