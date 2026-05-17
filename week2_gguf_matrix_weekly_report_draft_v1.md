# Week 2 GGUF Matrix Report Draft

## 1. Мета тижня

На другому тижні було розширено MVP benchmark pipeline для дипломного дослідження про вплив квантизації на якість і ресурсну ефективність локального запуску Phi-4-mini на українських текстових задачах.

Основне методологічне уточнення: попереднє порівняння `Transformers BF16 CPU` проти `Ollama Q4 CPU` було корисним як practical deployment baseline, але воно змішувало precision і runtime backend. Тому для чистішого експериментального ядра було сформовано GGUF/Ollama CPU-only матрицю.

## 2. Зафіксована матриця моделей

- `BF16` від professorf використано як external high-precision GGUF reference.
- `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M` від bartowski використано як основну same-runtime quantization curve.
- Усі запуски виконано в CPU-only режимі через Ollama з `num_gpu=0`.

## 3. Реалізовані компоненти

- Додано model registry у `config.py`.
- Додано перемикання активної моделі через `ACTIVE_OLLAMA_MODEL_KEY`.
- Реалізовано runner `run_week2_gguf_matrix.py`, який послідовно запускає всі моделі.
- Розширено QA та BELEBELE evaluation scripts метаданими про model source, quantization pipeline, imatrix і роль моделі.
- Додано resource monitoring: model process RSS, client process RSS, system memory peak, wall time, load time, prompt throughput, generation throughput.

## 4. QA results: UA-SQuAD 100 vs SQuAD 100

| quantization   |   uk_f1 |   en_f1 |   en_minus_uk_f1 |   uk_em |   en_em |   avg_model_peak_rss_mb |   memory_reduction_vs_bf16_pct |   uk_quality_retained_vs_q8_pct |   en_quality_retained_vs_q8_pct |
|:---------------|--------:|--------:|-----------------:|--------:|--------:|------------------------:|-------------------------------:|--------------------------------:|--------------------------------:|
| BF16           |  0.5143 |  0.8797 |           0.3654 |    0.33 |    0.8  |                 8065.43 |                           0    |                           91.64 |                           95.48 |
| Q8_0           |  0.5612 |  0.9213 |           0.3601 |    0.32 |    0.85 |                 4672.42 |                          42.07 |                          100    |                          100    |
| Q6_K           |  0.5664 |  0.8887 |           0.3223 |    0.31 |    0.78 |                 3786.59 |                          53.05 |                          100.93 |                           96.46 |
| Q5_K_M         |  0.5625 |  0.8887 |           0.3262 |    0.32 |    0.78 |                 3493.93 |                          56.68 |                          100.23 |                           96.46 |
| Q4_K_M         |  0.5563 |  0.8939 |           0.3376 |    0.33 |    0.79 |                 3155.93 |                          60.87 |                           99.13 |                           97.03 |

Ключове спостереження: на QA Q4_K_M зменшує model peak RSS з приблизно 8065.43 MB до 3155.93 MB, тобто на 60.87% відносно BF16 reference. Водночас якість на 100-sample subset залишається конкурентною щодо Q8_0: для української F1 retention становить 99.13%, для англійської — 97.03%.

## 5. BELEBELE results: ukr_Cyrl 100 vs eng_Latn 100

| quantization   |   uk_accuracy |   en_accuracy |   en_minus_uk_accuracy |   uk_invalid_rate |   en_invalid_rate |   avg_model_peak_rss_mb |   memory_reduction_vs_bf16_pct |   uk_quality_retained_vs_q8_pct |   en_quality_retained_vs_q8_pct |
|:---------------|--------------:|--------------:|-----------------------:|------------------:|------------------:|------------------------:|-------------------------------:|--------------------------------:|--------------------------------:|
| BF16           |          0.56 |          0.85 |                   0.29 |              0.01 |              0    |                 8102.85 |                           0    |                           81.16 |                           92.39 |
| Q8_0           |          0.69 |          0.92 |                   0.23 |              0.02 |              0.01 |                 4669.43 |                          42.37 |                          100    |                          100    |
| Q6_K           |          0.72 |          0.93 |                   0.21 |              0.02 |              0    |                 3779.68 |                          53.35 |                          104.35 |                          101.09 |
| Q5_K_M         |          0.71 |          0.92 |                   0.21 |              0    |              0    |                 3487.77 |                          56.96 |                          102.9  |                          100    |
| Q4_K_M         |          0.75 |          0.92 |                   0.17 |              0.01 |              0    |                 3147.39 |                          61.16 |                          108.7  |                          100    |

Ключове спостереження: на BELEBELE Q4_K_M також суттєво зменшує memory footprint і не демонструє деградації на цьому 100-sample subset. Українська accuracy для Q4_K_M становить 0.7500, англійська accuracy — 0.9200.

## 6. Language gap

| benchmark   | model_key        | quantization   | metric   |   uk_score |   en_score |   en_minus_uk |
|:------------|:-----------------|:---------------|:---------|-----------:|-----------:|--------------:|
| QA          | professorf_bf16  | BF16           | F1       |     0.5143 |     0.8797 |        0.3654 |
| QA          | professorf_bf16  | BF16           | EM       |     0.33   |     0.8    |        0.47   |
| QA          | bartowski_q8_0   | Q8_0           | F1       |     0.5612 |     0.9213 |        0.3601 |
| QA          | bartowski_q8_0   | Q8_0           | EM       |     0.32   |     0.85   |        0.53   |
| QA          | bartowski_q6_k   | Q6_K           | F1       |     0.5664 |     0.8887 |        0.3223 |
| QA          | bartowski_q6_k   | Q6_K           | EM       |     0.31   |     0.78   |        0.47   |
| QA          | bartowski_q5_k_m | Q5_K_M         | F1       |     0.5625 |     0.8887 |        0.3262 |
| QA          | bartowski_q5_k_m | Q5_K_M         | EM       |     0.32   |     0.78   |        0.46   |
| QA          | bartowski_q4_k_m | Q4_K_M         | F1       |     0.5563 |     0.8939 |        0.3376 |
| QA          | bartowski_q4_k_m | Q4_K_M         | EM       |     0.33   |     0.79   |        0.46   |
| BELEBELE    | professorf_bf16  | BF16           | accuracy |     0.56   |     0.85   |        0.29   |
| BELEBELE    | bartowski_q8_0   | Q8_0           | accuracy |     0.69   |     0.92   |        0.23   |
| BELEBELE    | bartowski_q6_k   | Q6_K           | accuracy |     0.72   |     0.93   |        0.21   |
| BELEBELE    | bartowski_q5_k_m | Q5_K_M         | accuracy |     0.71   |     0.92   |        0.21   |
| BELEBELE    | bartowski_q4_k_m | Q4_K_M         | accuracy |     0.75   |     0.92   |        0.17   |

На обох benchmark-ах зберігається стабільний розрив між англійською та українською. Це підтримує актуальність основного research question: чи впливає квантизація на українські задачі інакше, ніж на англійські контрольні задачі.

## 7. Проблеми та рішення

1. Було виявлено, що вимірювання пам’яті тільки Python-процесу не відображає пам’ять самої Ollama-моделі. Рішення: додано ResourceMonitor, який відстежує Ollama/llama/runner процеси.
2. Було виявлено, що одночасно завантажені Ollama-моделі спотворюють peak RSS. Рішення: runner перед кожною моделлю виконує `ollama stop` для відомих моделей.
3. Було виявлено, що direct HF tag для деяких моделей, наприклад Q3, може давати 404. Рішення: Q3 винесено в optional future step через manual Modelfile.
4. BF16 взято з professorf, а основну quantization curve — з bartowski. Рішення: BF16 позначено як external reference, а не як частину Bartowski same-runtime curve.

## 8. План на тиждень 3

- Зібрати графіки quality/resource trade-off: F1/accuracy vs model peak RSS, wall time vs quantization level.
- Додати третій benchmark: UAlign або Eval-UA-tion subset.
- Перевірити результати на більшому subset або повторити частину запусків для стабільності.
- Підготувати перший варіант методологічного розділу дипломної роботи.
