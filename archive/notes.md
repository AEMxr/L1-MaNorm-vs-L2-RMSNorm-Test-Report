# Stats
Both RMS and MA contain 202 operations as the dynamo results show.  Every curve csv has its corresponding png plot for easy readability. the text file, norm_runs_report_20260422.txt, gives a more legible general report on transformer model configurations, final train loss, final validation loss, step speed, and memory use:
```
Summary table

norm               n             train               val          best_val      best_step              grad     tr_step/s      tr_ms/st      tr_seq/s      tr_tok/s     tr_us/tok     ev_step/s      ev_ms/st      ev_seq/s      ev_tok/s     ev_us/tok          alloc_mb       reserved_mb   overfit_step
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
meanabs           30     0.8217±0.0780     1.5059±0.0424     1.4022±0.0242   3716.7±459.8     0.5726±0.0397     3.83±0.03   261.07±2.23      15.3±0.1 15690.4±133.3  63.738±0.543    10.96±0.16    91.28±1.34      43.8±0.6 44882.1±654.5  22.285±0.328       10300.4±0.0       10436.0±0.0   4596.2±367.4
meanabs_nocorr    30     1.4242±0.1488     1.6469±0.1057     1.6602±0.1045   4966.7±124.7     0.7292±0.0879     3.89±0.04   257.18±2.70      15.6±0.2 15928.6±164.1  62.787±0.659    11.11±0.10    89.99±0.81      44.5±0.4 45520.6±401.6  21.970±0.197       10300.4±0.0       10436.0±0.0            n/a
rms               30     0.7702±0.0832     1.5323±0.0362     1.3984±0.0239   3516.7±474.0     0.5890±0.0420     3.88±0.05   257.82±3.39      15.5±0.2 15889.8±206.0  62.944±0.826    11.10±0.15    90.13±1.21      44.4±0.6 45455.7±602.4  22.003±0.296       10299.9±0.0       10788.0±0.0   4534.5±434.1
``` 
### Average results

All values below are the reported mean ± std across **30 seeds**.

| Metric         |         MeanAbs |             RMS |  MeanAbs_NoCorr |
| -------------- | --------------: | --------------: | --------------: |
| Train loss     | 0.8217 ± 0.0780 | 0.7702 ± 0.0832 | 1.4242 ± 0.1488 |
| Val loss       | 1.5059 ± 0.0424 | 1.5323 ± 0.0362 | 1.6469 ± 0.1057 |
| Best val loss  | 1.4022 ± 0.0242 | 1.3984 ± 0.0239 | 1.6602 ± 0.1045 |
| Best step      |  3716.7 ± 459.8 |  3516.7 ± 474.0 |  4966.7 ± 124.7 |
| Grad norm      | 0.5726 ± 0.0397 | 0.5890 ± 0.0420 | 0.7292 ± 0.0879 |
| Train step/s   |     3.83 ± 0.03 |     3.88 ± 0.05 |     3.89 ± 0.04 |
| Train ms/step  |   261.07 ± 2.23 |   257.82 ± 3.39 |   257.18 ± 2.70 |
| Train tok/s    | 15690.4 ± 133.3 | 15889.8 ± 206.0 | 15928.6 ± 164.1 |
| Train µs/token |  63.738 ± 0.543 |  62.944 ± 0.826 |  62.787 ± 0.659 |
| Eval step/s    |    10.96 ± 0.16 |    11.10 ± 0.15 |    11.11 ± 0.10 |
| Eval ms/step   |    91.28 ± 1.34 |    90.13 ± 1.21 |    89.99 ± 0.81 |
| Eval tok/s     | 44882.1 ± 654.5 | 45455.7 ± 602.4 | 45520.6 ± 401.6 |
| Eval µs/token  |  22.285 ± 0.328 |  22.003 ± 0.296 |  21.970 ± 0.197 |
| Alloc MB       |   10300.4 ± 0.0 |   10299.9 ± 0.0 |   10300.4 ± 0.0 |
| Reserved MB    |   10436.0 ± 0.0 |   10788.0 ± 0.0 |   10436.0 ± 0.0 |
| Overfit step   |  4596.2 ± 367.4 |  4534.5 ± 434.1 |             n/a |

### MA vs RMS percentage comparisons

Below, a positive value means **MeanAbs is better** on that metric. For losses and latency, lower is better. For throughput and steps, higher is better. These comparisons are computed from the reported averages above.

| Metric         |                MeanAbs vs RMS |
| -------------- | ----------------------------: |
| Val loss       |             **+1.72%** better |
| Best val loss  |              **-0.27%** worse |
| Train loss     |              **-6.69%** worse |
| Grad norm      |              **+2.78%** lower |
| Best step      |              **+5.69%** later |
| Overfit step   |              **+1.36%** later |
| Train step/s   |             **-1.29%** slower |
| Train ms/step  |              **-1.26%** worse |
| Train tok/s    |             **-1.25%** slower |
| Train µs/token |              **-1.26%** worse |
| Eval step/s    |             **-1.26%** slower |
| Eval ms/step   |              **-1.28%** worse |
| Eval tok/s     |             **-1.26%** slower |
| Eval µs/token  |              **-1.28%** worse |
| Alloc MB       |              essentially tied |
| Reserved MB    | **+3.26%** better for MeanAbs |

### Absolute MA vs RMS differences

Sometimes the raw gap is easier to read than percentages. These are straight average-minus-average differences.

| Metric          | MeanAbs - RMS |
| --------------- | ------------: |
| Train loss      |       +0.0515 |
| Val loss        |   **-0.0264** |
| Best val loss   |       +0.0038 |
| Best step       |  +200.0 steps |
| Grad norm       |   **-0.0164** |
| Train tok/s     |  -199.4 tok/s |
| Eval tok/s      |  -573.6 tok/s |
| Train µs/token  |        +0.794 |
| Eval µs/token   |        +0.282 |
| Reserved memory | **-352.0 MB** |

### Relevant readout

MeanAbs is not winning cleanly across the board. It gets a **better final validation loss average** than RMS, and it also uses **less reserved memory** while showing a slightly **lower gradient norm** and slightly **later overfit onset** on average. But RMS still has the edge on **train loss**, **best validation loss**, and basically all the **speed/throughput** metrics.

The practical interpretation is: for this run, **MeanAbs looks slightly more favorable on average validation at the end of training**, but **RMS still looks a bit cleaner overall** because it trains faster, reaches a slightly better best checkpoint on average, and has lower training loss. The gap on best validation is very small: `1.4022` for MeanAbs vs `1.3984` for RMS, which is only about **0.27%** in RMS’s favor. The end-of-run validation gap goes the other way: `1.5059` for MeanAbs vs `1.5323` for RMS, about **1.72%** in MeanAbs’s favor.  Kernel and other ecosystem optimizations may improve MA validation loss including a differently weighted C value rather than the current Gaussian correction, and taking a less operation step approach for both forward and backward passes.

`meanabs_nocorr` is the obvious loser here. It is much worse on train loss, val loss, and best val loss, even though it is marginally faster than both. That speed edge is too small to matter given the quality drop.  The Gaussian or similar correction is needed for competitive results.

Fun thought, L1 variants of RMSNorm have the potential to be better (validation loss, speed, memory) with proper environment configurations.  A 352MB VRAM savings may justify the customizations. 
