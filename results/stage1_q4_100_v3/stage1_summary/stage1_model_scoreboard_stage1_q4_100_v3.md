# Stage 1 model scoreboard: stage1_q4_100_v3

## Scope

This report aggregates QA, BELEBELE and UAlign 100-sample summary files into one model-level scoreboard.

## Ranking logic

`avg_quality` is the mean of:

- QA Ukrainian F1
- QA English F1
- BELEBELE Ukrainian accuracy
- BELEBELE English accuracy
- UAlign Ethics Ukrainian accuracy
- UAlign Ethics English accuracy
- UAlign Social Chemistry Ukrainian accuracy
- UAlign Social Chemistry English accuracy

`en_minus_uk_quality_gap` is:

English average quality - Ukrainian average quality.

Positive values mean English performed better on average.

## Scoreboard

 quality_rank                               display_name    reasoning_mode  avg_quality  uk_quality_avg  en_quality_avg  en_minus_uk_quality_gap  avg_wall_time_sec  avg_eval_count  avg_model_process_peak_rss_mb  avg_invalid_answer_rate
            1                       Gemma-3-4B-it Q4_K_M          instruct       0.7501          0.6842          0.8160                   0.1318             4.7093          3.7683                      9720.0637                   0.0000
            2                          Qwen3.5-4B Q4_K_M default_or_hybrid       0.7361          0.6487          0.8236                   0.1749             6.1500          3.9317                     10858.5000                   0.0000
            3    NVIDIA-Nemotron-3-Nano-4B Q4_K_M /think             think       0.7265          0.6213          0.8317                   0.2104            23.0917         76.8300                     11100.7262                   0.0050
            4 NVIDIA-Nemotron-3-Nano-4B Q4_K_M /no_think          no_think       0.6971          0.5815          0.8127                   0.2312             8.7226          3.1967                     11263.6688                   0.0000
            5                 Phi-4-mini-instruct Q4_K_M          instruct       0.6898          0.6120          0.7676                   0.1556             3.7214          3.2467                      6895.7087                   0.0017
            6                         MiniCPM3-4B Q4_K_M          instruct       0.6881          0.5891          0.7870                   0.1979             4.8892          4.2900                     12110.0713                   0.0000
            7          MamayLM-Gemma-3-4B-IT-v1.0 Q4_K_M          instruct       0.6602          0.5863          0.7341                   0.1478             4.6144          3.0900                      8897.5987                   0.0000
            8          Hunyuan-4B-Instruct Q4_K_M /think             think       0.6535          0.5272          0.7799                   0.2527            34.6753        204.9450                     10120.4738                   0.0133
            9       Hunyuan-4B-Instruct Q4_K_M /no_think          no_think       0.6030          0.5073          0.6988                   0.1915             4.2913          6.2350                      8400.0225                   0.0000
           10              Qwen3-4B-Thinking-2507 Q4_K_M    reasoning_only       0.5328          0.5500          0.5157                  -0.0343           100.7513        739.6383                     10835.8350                   0.4267
           11              llm-jp-3-3.7b-instruct Q4_K_M          instruct       0.4099          0.2961          0.5237                   0.2276             3.7468          5.4600                     10570.8875                   0.0333
           12              Qwen3-4B-Instruct-2507 Q4_K_M          instruct       0.1345          0.1445          0.1244                  -0.0201             4.0392          3.1483                      8759.9313                   1.0000
