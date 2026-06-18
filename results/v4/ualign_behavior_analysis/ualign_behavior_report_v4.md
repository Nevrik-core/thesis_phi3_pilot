# UAlign Behavior Analysis Report

## 1. Scope

This report analyzes UAlign behavior beyond accuracy.

The goal is to check whether quantization changes the model's label-selection behavior on ethics/social-norm tasks:

- predicted label distribution;
- behavior/orientation drift between quantization modes;
- BF16 -> Q8_0 -> Q4_K_M answer changes;
- cross-lingual EN-UA stability on parallel examples.

Important: this does not measure the model's "true morality". It measures operational behavior under fixed prompts, fixed labels, and the UAlign label taxonomy.

## 2. Input data

Rows loaded: 20000

Quantization modes:
BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M

Tasks:
ethics, social_chemistry

Languages:
en, uk

## 3. Orientation summary

Interpretation:

- ETHICS: higher `ethics_unacceptable_rate` means the model more often classifies situations as unacceptable.
- Social Chemistry: higher `social_bad_rate` means more negative judgment; higher `social_good_rate` means more positive judgment.
- `mean_predicted_label_valid_delta_vs_BF16` shows the direction of drift compared with BF16.

quantization_name             task lang    n  accuracy  macro_f1_valid_only  invalid_rate  mean_predicted_label_valid  ethics_unacceptable_rate  social_bad_rate  social_neutral_rate  social_good_rate  accuracy_delta_vs_BF16  mean_predicted_label_valid_delta_vs_BF16
             BF16           ethics   en 1000     0.629               0.5893           0.0                       0.827                     0.827              NaN                  NaN               NaN                   0.000                                     0.000
             Q8_0           ethics   en 1000     0.771               0.7685           0.0                       0.619                     0.619              NaN                  NaN               NaN                   0.142                                    -0.208
             Q6_K           ethics   en 1000     0.742               0.7352           0.0                       0.676                     0.676              NaN                  NaN               NaN                   0.113                                    -0.151
           Q5_K_M           ethics   en 1000     0.763               0.7595           0.0                       0.637                     0.637              NaN                  NaN               NaN                   0.134                                    -0.190
           Q4_K_M           ethics   en 1000     0.737               0.7296           0.0                       0.681                     0.681              NaN                  NaN               NaN                   0.108                                    -0.146
             BF16           ethics   uk 1000     0.498               0.3719           0.0                       0.964                     0.964              NaN                  NaN               NaN                   0.000                                     0.000
             Q8_0           ethics   uk 1000     0.499               0.3711           0.0                       0.967                     0.967              NaN                  NaN               NaN                   0.001                                     0.003
             Q6_K           ethics   uk 1000     0.497               0.3642           0.0                       0.973                     0.973              NaN                  NaN               NaN                  -0.001                                     0.009
           Q5_K_M           ethics   uk 1000     0.490               0.3454           0.0                       0.986                     0.986              NaN                  NaN               NaN                  -0.008                                     0.022
           Q4_K_M           ethics   uk 1000     0.543               0.4644           0.0                       0.899                     0.899              NaN                  NaN               NaN                   0.045                                    -0.065
             BF16 social_chemistry   en 1000     0.620               0.5377           0.0                       0.683                       NaN            0.345                0.627             0.028                   0.000                                     0.000
             Q8_0 social_chemistry   en 1000     0.656               0.6452           0.0                       0.850                       NaN            0.328                0.494             0.178                   0.036                                     0.167
             Q6_K social_chemistry   en 1000     0.628               0.5644           0.0                       0.728                       NaN            0.326                0.620             0.054                   0.008                                     0.045
           Q5_K_M social_chemistry   en 1000     0.630               0.5979           0.0                       0.791                       NaN            0.326                0.557             0.117                   0.010                                     0.108
           Q4_K_M social_chemistry   en 1000     0.612               0.5067           0.0                       0.695                       NaN            0.307                0.691             0.002                  -0.008                                     0.012
             BF16 social_chemistry   uk 1000     0.561               0.4714           0.0                       1.136                       NaN            0.431                0.002             0.567                   0.000                                     0.000
             Q8_0 social_chemistry   uk 1000     0.573               0.4887           0.0                       1.157                       NaN            0.416                0.011             0.573                   0.012                                     0.021
             Q6_K social_chemistry   uk 1000     0.585               0.5193           0.0                       1.121                       NaN            0.420                0.039             0.541                   0.024                                    -0.015
           Q5_K_M social_chemistry   uk 1000     0.586               0.5195           0.0                       1.157                       NaN            0.404                0.035             0.561                   0.025                                     0.021
           Q4_K_M social_chemistry   uk 1000     0.568               0.4910           0.0                       1.131                       NaN            0.424                0.021             0.555                   0.007                                    -0.005

## 4. Quantization flips summary

Interpretation:

- `label_changed_rate` shows how often the predicted label changes between two quantization modes.
- `correct_to_wrong_rate` is potential degradation hidden by aggregate metrics.
- `wrong_to_correct_rate` is potential improvement or noise.
- Directional columns show whether labels became stricter/more permissive or more positive/negative.

reference_quantization target_quantization             task lang    n  label_changed_rate  accuracy_delta  correct_to_wrong_rate  wrong_to_correct_rate  became_more_strict_or_unacceptable  became_more_permissive_or_acceptable  became_more_positive  became_more_negative
                  BF16              Q4_K_M           ethics   en 1000               0.152           0.108                  0.022                  0.130                                 3.0                                 149.0                   0.0                   0.0
                  BF16              Q4_K_M           ethics   uk 1000               0.093           0.045                  0.024                  0.069                                14.0                                  79.0                   0.0                   0.0
                  BF16              Q4_K_M social_chemistry   en 1000               0.079          -0.008                  0.040                  0.032                                 0.0                                   0.0                  46.0                  33.0
                  BF16              Q4_K_M social_chemistry   uk 1000               0.078           0.007                  0.017                  0.024                                 0.0                                   0.0                  35.0                  43.0
                  BF16                Q8_0           ethics   en 1000               0.210           0.142                  0.034                  0.176                                 1.0                                 209.0                   0.0                   0.0
                  BF16                Q8_0           ethics   uk 1000               0.037           0.001                  0.018                  0.019                                20.0                                  17.0                   0.0                   0.0
                  BF16                Q8_0 social_chemistry   en 1000               0.186           0.036                  0.070                  0.106                                 0.0                                   0.0                 174.0                  12.0
                  BF16                Q8_0 social_chemistry   uk 1000               0.062           0.012                  0.008                  0.020                                 0.0                                   0.0                  35.0                  27.0
                  Q8_0              Q4_K_M           ethics   en 1000               0.064          -0.034                  0.049                  0.015                                63.0                                   1.0                   0.0                   0.0
                  Q8_0              Q4_K_M           ethics   uk 1000               0.074           0.044                  0.015                  0.059                                 3.0                                  71.0                   0.0                   0.0
                  Q8_0              Q4_K_M social_chemistry   en 1000               0.202          -0.044                  0.122                  0.078                                 0.0                                   0.0                  24.0                 178.0
                  Q8_0              Q4_K_M social_chemistry   uk 1000               0.041          -0.005                  0.014                  0.009                                 0.0                                   0.0                  11.0                  30.0

## 5. EN-UA consistency summary

Interpretation:

- `label_match_rate_on_valid_pairs` shows how often Ukrainian and English versions of the same example receive the same valid label.
- A drop after quantization suggests lower cross-lingual behavioral stability.
- `en_minus_uk_accuracy` shows whether English remains easier than Ukrainian for this task.

quantization_name             task    n  both_valid_rate  label_match_rate_on_valid_pairs  raw_label_match_rate  uk_accuracy  en_accuracy  en_minus_uk_accuracy  mean_en_minus_uk_label  label_match_rate_on_valid_pairs_delta_vs_BF16
             BF16           ethics 1000              1.0                            0.825                 0.825        0.498        0.629                 0.131                  -0.137                                          0.000
             Q8_0           ethics 1000              1.0                            0.630                 0.630        0.499        0.771                 0.272                  -0.348                                         -0.195
             Q6_K           ethics 1000              1.0                            0.687                 0.687        0.497        0.742                 0.245                  -0.297                                         -0.138
           Q5_K_M           ethics 1000              1.0                            0.641                 0.641        0.490        0.763                 0.273                  -0.349                                         -0.184
           Q4_K_M           ethics 1000              1.0                            0.728                 0.728        0.543        0.737                 0.194                  -0.218                                         -0.097
             BF16 social_chemistry 1000              1.0                            0.356                 0.356        0.561        0.620                 0.059                  -0.453                                          0.000
             Q8_0 social_chemistry 1000              1.0                            0.489                 0.489        0.573        0.656                 0.083                  -0.307                                          0.133
             Q6_K social_chemistry 1000              1.0                            0.398                 0.398        0.585        0.628                 0.043                  -0.393                                          0.042
           Q5_K_M social_chemistry 1000              1.0                            0.448                 0.448        0.586        0.630                 0.044                  -0.366                                          0.092
           Q4_K_M social_chemistry 1000              1.0                            0.313                 0.313        0.568        0.612                 0.044                  -0.436                                         -0.043

## 6. Main output files

- `ualign_label_distribution_v4.csv`
- `ualign_gold_distribution_v4.csv`
- `ualign_orientation_summary_v4.csv`
- `ualign_confusion_matrix_long_v4.csv`
- `ualign_quantization_flips_v4.csv`
- `ualign_quantization_flips_summary_v4.csv`
- `ualign_en_uk_consistency_v4.csv`
- `ualign_en_uk_consistency_summary_v4.csv`
- `ualign_behavior_inspection_cases_v4.csv`

