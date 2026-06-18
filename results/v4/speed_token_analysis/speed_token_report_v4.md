# Speed and Token Efficiency Analysis v4

## 1. Scope

This report separates practical latency from internal token throughput.

- `wall_time_sec` is interpreted as end-to-end latency per sample.
- `prompt_tokens_per_sec` measures prompt/context processing speed.
- `generation_tokens_per_sec` measures answer generation speed and is used as the closest available proxy for "cleaner" model generation speed in the Ollama/GGUF setup.
- token counts are used as an additional resource-cost characteristic.

## 2. Why this layer is needed

Memory usage alone is not enough to describe resource efficiency. A quantized model may use less RAM but not necessarily produce lower end-to-end latency. Therefore, latency, throughput, and token cost are analyzed separately.

## 3. Main compact summary

benchmark             task lang quantization_name    n quality_metric  quality_score  avg_wall_time_sec  avg_prompt_tokens_per_sample  avg_generated_tokens_per_sample  avg_total_tokens_per_sample  avg_prompt_tokens_per_sec  avg_generation_tokens_per_sec  avg_model_process_peak_rss_mb
 BELEBELE         BELEBELE   en              BF16  900       accuracy         0.8211             4.5195                      197.4211                           2.3122                     199.7333                   139.8465                         6.8741                        8350.03
 BELEBELE         BELEBELE   en              Q8_0  900       accuracy         0.8844             5.0658                      197.4211                           3.4456                     200.8667                   109.8710                         9.8348                        4815.20
 BELEBELE         BELEBELE   en              Q6_K  900       accuracy         0.8878             5.3920                      197.4211                           3.4622                     200.8833                    95.1079                        12.8433                        4019.65
 BELEBELE         BELEBELE   en            Q5_K_M  900       accuracy         0.8867             5.4905                      197.4211                           3.6911                     201.1122                    92.1244                        12.8513                        3611.21
 BELEBELE         BELEBELE   en            Q4_K_M  900       accuracy         0.8833             5.1113                      197.4211                           3.7300                     201.1511                   104.3083                        15.0159                        3271.80
 BELEBELE         BELEBELE   uk              BF16  900       accuracy         0.6111             6.3146                      345.8456                           3.3444                     349.1900                   141.3355                         5.4305                        8339.17
 BELEBELE         BELEBELE   uk              Q8_0  900       accuracy         0.7356             6.7784                      345.8456                           2.8200                     348.6656                   112.8269                        10.6811                        4807.55
 BELEBELE         BELEBELE   uk              Q6_K  900       accuracy         0.7456             7.5440                      345.8456                           2.9200                     348.7656                    96.4569                        13.6360                        3922.54
 BELEBELE         BELEBELE   uk            Q5_K_M  900       accuracy         0.7356             7.6699                      345.8456                           3.1411                     348.9867                    94.1947                        13.8266                        3602.71
 BELEBELE         BELEBELE   uk            Q4_K_M  900       accuracy         0.7433             7.1396                      345.8456                           3.3967                     349.2422                   104.4758                        14.6607                        3263.05
       QA               QA   en              BF16 1000             F1         0.8267             4.1306                      186.1640                           5.1450                     191.3090                   429.4578                         4.9122                        8349.40
       QA               QA   en              Q8_0 1000             F1         0.8644             3.6963                      186.1640                           5.0210                     191.1850                   377.7773                         9.1423                        4822.94
       QA               QA   en              Q6_K 1000             F1         0.8586             3.6413                      186.1640                           4.9900                     191.1540                   362.1784                        11.7989                        3895.59
       QA               QA   en            Q5_K_M 1000             F1         0.8588             3.6251                      186.1640                           5.0150                     191.1790                   356.2284                        12.5871                        3662.72
       QA               QA   en            Q4_K_M 1000             F1         0.8624             3.4867                      186.1640                           4.9890                     191.1530                   394.5361                        14.0666                        3282.12
       QA               QA   uk              BF16 1000             F1         0.5166             6.4003                      299.8210                           9.9730                     309.7940                   399.2804                         4.3619                        8332.16
       QA               QA   uk              Q8_0 1000             F1         0.5757             5.6955                      299.8210                           9.8800                     309.7010                   321.3392                         7.6610                        4809.77
       QA               QA   uk              Q6_K 1000             F1         0.5748             5.7518                      299.8210                           9.8830                     309.7040                   288.3696                         9.6486                        3879.59
       QA               QA   uk            Q5_K_M 1000             F1         0.5720             5.6656                      299.8210                           9.8320                     309.6530                   285.4091                        10.7675                        3724.00
       QA               QA   uk            Q4_K_M 1000             F1         0.5677             5.3409                      299.8210                           9.6190                     309.4400                   313.0301                        11.7759                        3268.01
   UAlign           ethics   en              BF16 1000       accuracy         0.6290             2.9250                       64.1710                           2.0000                      66.1710                   182.1100                         7.0359                        8355.33
   UAlign           ethics   en              Q8_0 1000       accuracy         0.7710             2.8427                       64.1710                           2.0000                      66.1710                   170.8867                        13.0948                        4834.71
   UAlign           ethics   en              Q6_K 1000       accuracy         0.7420             2.8203                       64.1710                           2.0000                      66.1710                   169.3836                        16.6383                        4039.58
   UAlign           ethics   en            Q5_K_M 1000       accuracy         0.7630             2.8159                       64.1710                           2.0000                      66.1710                   167.8621                        18.1511                        3632.74
   UAlign           ethics   en            Q4_K_M 1000       accuracy         0.7370             2.7586                       64.1710                           2.0000                      66.1710                   187.6061                        20.1682                        3291.23
   UAlign           ethics   uk              BF16 1000       accuracy         0.4980             3.1316                      101.0130                           2.0000                     103.0130                   184.5545                         7.2289                        8360.27
   UAlign           ethics   uk              Q8_0 1000       accuracy         0.4990             3.1739                      101.0130                           2.0000                     103.0130                   145.7539                        13.0152                        4830.39
   UAlign           ethics   uk              Q6_K 1000       accuracy         0.4970             3.2263                      101.0130                           2.0000                     103.0130                   131.8003                        16.5183                        4033.73
   UAlign           ethics   uk            Q5_K_M 1000       accuracy         0.4900             3.2283                      101.0130                           2.0000                     103.0130                   129.8569                        18.0893                        3626.40
   UAlign           ethics   uk            Q4_K_M 1000       accuracy         0.5430             3.1285                      101.0130                           2.0000                     103.0130                   144.7857                        20.1745                        3285.91
   UAlign social_chemistry   en              BF16 1000       accuracy         0.6200             2.8810                       67.3430                           2.0000                      69.3430                   214.4208                         7.0368                        8263.21
   UAlign social_chemistry   en              Q8_0 1000       accuracy         0.6560             2.7559                       67.3430                           2.0000                      69.3430                   218.7704                        13.1051                        4793.65
   UAlign social_chemistry   en              Q6_K 1000       accuracy         0.6280             2.7120                       67.3430                           2.0000                      69.3430                   225.9106                        16.6440                        4040.05
   UAlign social_chemistry   en            Q5_K_M 1000       accuracy         0.6300             2.7040                       67.3430                           2.0000                      69.3430                   224.7561                        18.1881                        3633.65
   UAlign social_chemistry   en            Q4_K_M 1000       accuracy         0.6120             2.6520                       67.3430                           2.0000                      69.3430                   253.4385                        20.8461                        3267.61
   UAlign social_chemistry   uk              BF16 1000       accuracy         0.5610             3.0224                       99.8880                           2.0000                     101.8880                   224.8980                         7.0276                        8325.47
   UAlign social_chemistry   uk              Q8_0 1000       accuracy         0.5730             2.9845                       99.8880                           2.0000                     101.8880                   189.3094                        13.3778                        4829.38
   UAlign social_chemistry   uk              Q6_K 1000       accuracy         0.5850             3.0131                       99.8880                           2.0000                     101.8880                   174.7565                        16.5565                        4036.01
   UAlign social_chemistry   uk            Q5_K_M 1000       accuracy         0.5860             3.0101                       99.8880                           2.0000                     101.8880                   172.5835                        18.1010                        3630.18
   UAlign social_chemistry   uk            Q4_K_M 1000       accuracy         0.5680             2.9159                       99.8880                           2.0000                     101.8880                   195.5271                        21.3396                        3248.54


## 4. Main output files

- `speed_token_summary_v4.csv`
- `speed_token_summary_with_deltas_v4.csv`
- `speed_token_compact_v4.csv`
- `speed_token_efficiency_v4.csv`
- `figures/01_generation_tokens_per_sec.png`
- `figures/02_prompt_tokens_per_sec.png`
- `figures/03_avg_total_tokens_per_sample.png`
- `figures/04_wall_time_vs_generation_tokens_per_sec.png`
- `figures/05_memory_vs_generation_tokens_per_sec.png`
- `figures/06_quality_vs_avg_total_tokens_per_sample.png`
