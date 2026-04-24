# MeanAbsNorm vs RMSNorm: A Transformer Normalization Ablation Study

## Abstract

This report presents an ablation study of three hidden-vector normalization methods in byte-level decoder-only transformer training: RMSNorm, corrected MeanAbsNorm, and uncorrected MeanAbsNorm-NoCorr. RMSNorm uses root-mean-square scaling, while MeanAbsNorm replaces the RMS statistic with mean absolute magnitude, optionally calibrated by the Gaussian correction constant \(C=\sqrt{\pi/2}\). The study measures how these scale statistics affect validation behavior, overfitting dynamics, throughput, memory use, and corpus sensitivity under controlled architecture, optimizer, data split, and seed conditions.

The current results cover two completed corpora, *The City of God* and *Anne of Green Gables*, with 30 random seeds per normalization method per corpus. On *The City of God*, RMSNorm achieves the lowest best validation loss and lowest average checkpoint validation loss, while MeanAbsNorm achieves lower final reported validation loss and a smaller overfitting gap. On *Anne of Green Gables*, MeanAbsNorm slightly outperforms RMSNorm across best validation loss, final reported validation loss, average checkpoint validation loss, overfitting gap, throughput, and reserved memory.

MeanAbsNorm-NoCorr shows the strongest corpus-dependent behavior among the tested variants. It performs poorly relative to RMSNorm and corrected MeanAbsNorm on *The City of God*, but shows unexpectedly strong late-validation behavior on *Anne of Green Gables*, despite worse training-loss and gradient-norm behavior. These results suggest that the choice of normalization scale statistic materially affects transformer training dynamics and that corpus structure may influence the observed tradeoffs between RMS-based, corrected mean-absolute, and uncorrected mean-absolute scaling.

## 1. Introduction

Normalization is a core component of transformer training. It affects the scale of hidden activations, the stability of gradient flow, the behavior of optimization, and the relationship between training loss and validation loss over time. In decoder-only transformer models, small changes to the normalization statistic can produce measurable differences in convergence behavior, throughput, memory use, and generalization dynamics.

This report studies three hidden-vector normalization variants under controlled byte-level transformer training conditions: RMSNorm, corrected MeanAbsNorm, and MeanAbsNorm-NoCorr. RMSNorm rescales each hidden vector by its root-mean-square magnitude. MeanAbsNorm instead rescales each hidden vector by its mean absolute magnitude, using a Gaussian correction constant to place the expected scale closer to RMS magnitude. MeanAbsNorm-NoCorr removes that correction constant and serves as an ablation of the correction itself.

The purpose of this study is to report the empirical behavior of these normalization variants across repeated training runs. The comparison is not limited to a single final loss value. Instead, the report tracks best validation loss, final reported validation loss, average validation loss across checkpoints, overfitting gap, training loss, throughput, reserved memory, and corpus-dependent behavior. This allows the normalization methods to be compared across both optimization behavior and practical runtime characteristics.

The current experimental set includes completed 30-seed runs on *The City of God* and *Anne of Green Gables*, with a Sherlock Holmes corpus section reserved for additional results. All tested methods are evaluated using the same model configuration, optimizer settings, tokenization method, train/validation split strategy, seed count, training duration, and evaluation interval. The normalization method and corpus are the primary variables under comparison.

## 2. Research Question and Contributions

This study examines how different hidden-vector scale statistics affect decoder-only transformer training behavior under controlled byte-level training conditions. The central research question is:

> How do RMS-based scaling, corrected mean-absolute scaling, and uncorrected mean-absolute scaling differ in validation behavior, overfitting dynamics, throughput, memory use, and corpus sensitivity when used as drop-in normalization variants in the same transformer architecture?

The study evaluates three normalization methods:

1. **RMSNorm**, which scales activations by root-mean-square magnitude.
2. **MeanAbsNorm**, which scales activations by mean absolute magnitude with a Gaussian correction constant.
3. **MeanAbsNorm-NoCorr**, which scales activations by mean absolute magnitude without the correction constant.

Contributions of this report:

- Controlled ablation of RMS-based, corrected mean-absolute, and uncorrected mean-absolute hidden-vector normalization.
- 30-seed evaluation per normalization method for each completed corpus.
- Multi-metric analysis covering best validation loss, final reported validation loss, average checkpoint validation loss, overfitting gap, training loss, throughput, and reserved GPU memory.
- Corpus-sensitive analysis using complete literary texts rather than isolated short samples.
- Separate evaluation of the Gaussian correction constant through the MeanAbsNorm-NoCorr ablation.
- Reproducible reporting structure linking aggregate results, per-seed results, validation curves, implementation details, and statistical summaries.

## 3. Related Work and Distinction

Normalization methods are widely used to stabilize neural network training by controlling activation scale. Layer Normalization was introduced as an alternative to Batch Normalization that computes normalization statistics from the hidden units within a layer for each individual training case, avoiding dependence on mini-batch statistics. Unlike Batch Normalization, LayerNorm applies the same computation during training and inference. :contentReference[oaicite:0]{index=0}

RMSNorm modifies LayerNorm by removing the re-centering operation and retaining only root-mean-square rescaling. Zhang and Sennrich proposed RMSNorm on the hypothesis that re-centering invariance is not always necessary, while rescaling invariance remains useful for training stability and efficiency. RMSNorm is therefore the primary baseline in this study because the tested MeanAbsNorm variants follow the same general no-mean-centering structure. :contentReference[oaicite:1]{index=1}

Prior work has also explored L1-style normalization. Hoffer et al. studied alternatives to L2 BatchNorm, including L1 BatchNorm and \(L_{\infty}\)-based normalization, with emphasis on computational efficiency, memory behavior, and numerical stability. That work is relevant because it replaces an L2-style statistic with an L1-style statistic, but its normalization setting differs from the one tested here. :contentReference[oaicite:2]{index=2}

This study does not evaluate BatchNorm and does not use batch statistics. The tested MeanAbsNorm variants normalize each token's hidden activation vector independently, following the same general placement pattern as RMSNorm. They do not subtract the mean, do not compute centered deviations, and do not aggregate statistics across a batch.

The distinction is therefore structural: prior L1 BatchNorm work studies mean-centered batch-normalization variants, while this study evaluates RMSNorm-style hidden-vector normalization using mean absolute magnitude as the scale statistic. The comparison is between RMS-based, corrected mean-absolute, and uncorrected mean-absolute rescaling within the same decoder-only transformer architecture.

## 4. Mathematical Definitions

### 4.1 Notation

Let \(x \in \mathbb{R}^d\) be the hidden activation vector normalized at a single token position, where \(d\) is the hidden dimension. Let \(\epsilon\) be a small numerical-stability constant, and let \(\gamma \in \mathbb{R}^d\) be the learned elementwise gain parameter.

All tested methods normalize over the hidden dimension of each token independently.

### 4.2 RMSNorm

RMSNorm rescales the hidden vector using its root-mean-square magnitude: 

\[ 
\operatorname{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon} 
\] 

\[ 
y_i = \gamma_i \frac{x_i}{\operatorname{RMS}(x)} 
\] 

Equivalently:

\[
y_i = \gamma_i \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2 + \epsilon}}
\]

### 4.3 MeanAbsNorm

MeanAbsNorm replaces the RMS scale statistic with mean absolute magnitude:

\[
\operatorname{MA}(x) = \frac{1}{d}\sum_{i=1}^{d}|x_i|
\]

The corrected MeanAbsNorm scale is:

\[
\operatorname{MeanAbsScale}(x) = C \cdot \operatorname{MA}(x) + \epsilon
\]

The normalized output is:

\[
y_i = \gamma_i \frac{x_i}{C \cdot \operatorname{MA}(x) + \epsilon}
\]

Equivalently:

\[
y_i = \gamma_i \frac{x_i}{C \cdot \frac{1}{d}\sum_{j=1}^{d}|x_j| + \epsilon}
\]

### 4.4 MeanAbsNorm-NoCorr

MeanAbsNorm-NoCorr removes the correction constant and uses the raw mean absolute magnitude as the scale statistic:

\[
\operatorname{NoCorrScale}(x) = \operatorname{MA}(x) + \epsilon
\]

The normalized output is:

\[
y_i = \gamma_i \frac{x_i}{\operatorname{MA}(x) + \epsilon}
\]

Equivalently:

\[
y_i = \gamma_i \frac{x_i}{\frac{1}{d}\sum_{j=1}^{d}|x_j| + \epsilon}
\]

### 4.5 Correction Constant

The corrected MeanAbsNorm variant uses:

\[
C = \sqrt{\frac{\pi}{2}}
\]

This constant comes from the expected absolute value of a standard normal variable. For:

\[
Z \sim \mathcal{N}(0,1)
\]

the expected absolute value is:

\[
\mathbb{E}[|Z|] = \sqrt{\frac{2}{\pi}}
\]

The reciprocal is:

\[
\frac{1}{\mathbb{E}[|Z|]} = \sqrt{\frac{\pi}{2}}
\]

Therefore, multiplying mean absolute magnitude by \(C\) approximately aligns the expected mean-absolute scale with unit RMS scale under a zero-mean Gaussian assumption:

\[
C \cdot \mathbb{E}[|Z|] = \sqrt{\frac{\pi}{2}} \cdot \sqrt{\frac{2}{\pi}} = 1
\]

### 4.6 Scale-Statistic Comparison


The three tested methods differ only in the scale statistic used in the denominator:

| Method | Scale Statistic | Normalized Form |
|---|---|---|
| RMSNorm | \(\sqrt{\operatorname{mean}(x^2) + \epsilon}\) | \(x / \sqrt{\operatorname{mean}(x^2) + \epsilon}\) |
| MeanAbsNorm | \(C \cdot \operatorname{mean}(|x|) + \epsilon\) | \(x / (C \cdot \operatorname{mean}(|x|) + \epsilon)\) |
| MeanAbsNorm-NoCorr | \(\operatorname{mean}(|x|) + \epsilon\) | \(x / (\operatorname{mean}(|x|) + \epsilon)\) |

All three tested methods:

- normalize each token hidden vector independently
- normalize across the hidden dimension
- do not use batch statistics
- do not subtract the mean
- use the same learned elementwise gain parameter \(\gamma\)
- differ only in the denominator scale statistic

## 5. Implementation Details

The tested normalization variants were implemented as drop-in replacements at the same normalization sites in the decoder-only transformer. Each variant receives the same hidden activation tensor, uses the same learned gain parameter shape, and returns a tensor with the same shape as the input. The surrounding attention, feedforward, residual, optimizer, and training code are unchanged.

### 5.1 Normalization Placement

Each normalization variant is applied at the same transformer normalization call sites used by the baseline RMSNorm configuration. For every run, the selected normalization module is substituted without changing the block structure, residual connections, attention mechanism, feedforward network, optimizer, scheduler, data loader, or evaluation logic.

All tested variants normalize over the final tensor dimension:
```python
dim=-1
```
For a hidden tensor shaped Like:
(batch, sequence_length, hidden_dim)
the normalization statistic is computed independently for each token position over hidden_dim.

The model uses a pre-norm decoder-only transformer block. Within each block, the selected normalization variant is applied before the causal self-attention sublayer and before the feedforward sublayer:
```python
x = x + self.attn(self.n1(x))
x = x + self.ff(self.n2(x))
```
The model also applies the same selected normalization variant as a final normalization after all transformer blocks and before the language-model head:
```python
x = self.final_norm(x)
logits = self.head(x)
```
For all tested normalization methods, the placement is unchanged. The only implementation difference is the scale statistic used inside the normalization module.

### 5.2 RMSNorm Implementation

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(mean_sq + self.eps)
        return x * inv * self.weight
```

### 5.3 MeanAbsNorm Implementation

```python
class MeanAbsNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.c = math.sqrt(math.pi / 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().mean(dim=-1, keepdim=True) * self.c
        return x / (scale + self.eps) * self.weight
```

### 5.4 MeanAbsNorm-NoCorr Implementation

```python
class MeanAbsNormNoCorrection(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().mean(dim=-1, keepdim=True)
        return x / (scale + self.eps) * self.weight
```

### 5.5 Controlled Implementation Conditions

All three tested normalization variants use the same:

- transformer architecture
- normalization call sites
- input and output tensor shapes
- hidden-dimension reduction axis, dim=-1
- learned elementwise gain parameter, weight
- default numerical-stability value, eps = 1e-5
- no learned bias parameter
- optimizer
- learning-rate schedule
- seed set
- data split
- training duration
- evaluation interval

The implementation-level change is isolated to the normalization scale computation.

## 6. Experimental Setup

### 6.1 Model Architecture

All experiments use the same decoder-only causal language model architecture. The model is trained as a byte-level next-token predictor with a vocabulary size of 256.

| Parameter | Value |
|---|---:|
| Vocabulary size | 256 |
| Sequence length / context length | 1024 |
| Model width, \(d_{\text{model}}\) | 704 |
| Attention heads | 22 |
| Head dimension | 32 |
| Transformer layers | 15 |
| Feedforward hidden size | 2816 |
| Dropout | 0.0 |
| Normalization epsilon | \(1 \times 10^{-5}\) |
| Parameter count | 90,314,048 |

The model uses learned token embeddings and learned positional embeddings. Each transformer block contains causal self-attention followed by a feedforward network. The feedforward network uses GELU activation with the `tanh` approximation. The attention projections, output projection, feedforward projections, and language-model head are implemented as linear layers without bias.

The model uses a pre-norm block structure. The selected normalization method is applied before the attention sublayer and before the feedforward sublayer, followed by residual addition. A final normalization layer is applied after the final transformer block and before the language-model head.

### 6.2 Training Configuration

All completed runs use the same training configuration.

| Parameter | Value |
|---|---:|
| Training steps | 5000 |
| Warmup steps | 50 |
| Batch size | 4 |
| Sequence length | 1024 |
| Learning rate | 0.0003 |
| Optimizer | AdamW |
| Gradient clipping | 1.0 |
| Evaluation interval | every 500 steps |
| Evaluation batches | 32 |
| Training-loss window | 100 steps |
| Compilation | disabled |
| Compile backend | none |
| Device | CUDA |

Each run performs 50 warmup steps before the measured training loop. CUDA peak-memory statistics are reset after warmup so that reported memory values correspond to the measured run rather than initialization or warmup artifacts.

### 6.3 Tokenization

The experiments use byte-level tokenization. Each input file is read as raw bytes, and each byte value is treated as one token in a vocabulary of size 256.

The training objective is next-byte prediction. For each sampled sequence \(x\), the target sequence \(y\) is the same byte stream shifted forward by one position:
```python id="4t7kwn"
x = source[s : s + seq_len]
y = source[s + 1 : s + seq_len + 1]
```
No BPE, unigram, word-level tokenizer, sentencepiece tokenizer, or language-specific preprocessing is used.

### 6.4 Train/Validation Split

Each corpus is split sequentially by byte order:

| Split          |            Portion |
| -------------- | -----------------: |
| Training set   | first 90% of bytes |
| Validation set | final 10% of bytes |

The split is not randomized. Training samples are drawn randomly from the training portion of the byte stream, and validation samples are drawn randomly from the validation portion.

This means validation loss measures performance on the later portion of the same complete text, rather than on a randomly shuffled subset of the full corpus.

### 6.5 Seeds

Each completed corpus uses 30 random seeds per normalization method:
```bash
-- seeds 598422775,1786548453,1426323417,736896387,1722602715,1639699922,1048955721,370076197,188292180,482242919,676750256,405379411,130604721,1310941852,1162914961,1109572572,1431649702,1497939373,741210827,129344154,273318888,1718616343,1970508508,336498026,1301334618,1258359901,2022903755,281919623,922641176,1225032379  
```
The same seed list is used for each normalization method within a corpus.

### 6.6 Controlled Variables

The completed experiments control the following variables across normalization methods:

- model architecture
- parameter count
- normalization placement
- optimizer
- learning rate
- gradient clipping value
- batch size
- sequence length
- training steps
- warmup steps
- evaluation interval
- evaluation batch count
- train/validation split method
- byte-level tokenization
- seed list
- device type
- compile setting

The primary experimental variable is the normalization scale statistic:

| Method             | Scale Statistic                     |
| ------------------ | ----------------------------------- |
| RMSNorm            | root-mean-square magnitude          |
| MeanAbsNorm        | corrected mean absolute magnitude   |
| MeanAbsNorm-NoCorr | uncorrected mean absolute magnitude |

The secondary experimental variable is corpus material.

## 7. Corpora

Each corpus is treated as a complete byte stream. Because the experiments use byte-level tokenization, every byte value is represented directly as one token in a vocabulary of size 256. No word-level, BPE, unigram, or sentencepiece tokenizer is used.

Each corpus is split sequentially by byte order, with the first 90% used for training and the final 10% used for validation. The validation set therefore measures performance on the later portion of the same complete text, not on a randomly shuffled subset.

| Corpus | Status | Total Bytes / Tokens | Train Bytes / Tokens | Validation Bytes / Tokens | Completed Runs |
|---|---:|---:|---:|---:|---:|
| *The City of God* | Complete | 1,357,439 | 1,221,695 | 135,744 | 90 |
| *Anne of Green Gables* | Complete | 598,286 | 538,457 | 59,829 | 90 |
| *The Adventures of Sherlock Holmes* | Reserved | pending | pending | pending | 0 |

Each completed corpus contains 30 runs for each tested normalization method: RMSNorm, MeanAbsNorm, and MeanAbsNorm-NoCorr.

### 7.1 The City of God

*The City of God* is the largest completed corpus in the current experiment set. The source text is a Project Gutenberg edition of *The City of God, Volume I* by Augustine of Hippo, translated and edited in a public-domain edition. The file includes front matter, title-page material, table-of-contents structure, editor’s preface, footnotes, book divisions, argument summaries, numbered sections, and long-form theological prose. :contentReference[oaicite:0]{index=0}

| Property | Value |
|---|---:|
| Total bytes / tokens | 1,357,439 |
| Training bytes / tokens | 1,221,695 |
| Validation bytes / tokens | 135,744 |
| Completed RMSNorm runs | 30 |
| Completed MeanAbsNorm runs | 30 |
| Completed MeanAbsNorm-NoCorr runs | 30 |
| Total completed runs | 90 |

Corpus characteristics relevant to byte-level transformer training:

- largest completed corpus by byte count
- formal theological and philosophical prose
- long argumentative paragraphs
- repeated book, chapter, section, and footnote patterns
- frequent capitalized headings and structural markers
- recurring abstract vocabulary
- comparatively stable rhetorical and syntactic structure
- lower dialogue density than the narrative-fiction corpora

Because this corpus is larger and more formally structured, it provides a different training regime from the shorter fiction corpora. The repeated headings, book divisions, footnotes, and formal prose patterns may provide more stable byte-level structure, while the long argumentative sentences may also increase long-range dependency demands.

### 7.2 Anne of Green Gables

*Anne of Green Gables* is the second completed corpus in the current experiment set. The source text is a Project Gutenberg edition of L. M. Montgomery’s novel. The file includes Project Gutenberg front matter, title and contents material, chapter headings, narrative prose, dialogue, character names, quoted speech, and descriptive scene writing. :contentReference[oaicite:1]{index=1}

| Property | Value |
|---|---:|
| Total bytes / tokens | 598,286 |
| Training bytes / tokens | 538,457 |
| Validation bytes / tokens | 59,829 |
| Completed RMSNorm runs | 30 |
| Completed MeanAbsNorm runs | 30 |
| Completed MeanAbsNorm-NoCorr runs | 30 |
| Total completed runs | 90 |

Corpus characteristics relevant to byte-level transformer training:

- smaller than *The City of God*
- narrative fiction prose
- frequent dialogue and quoted speech
- recurring character names and place names
- conversational punctuation patterns
- chapter-based structure
- more emotional and tonal variation than *The City of God*
- less formal argumentative repetition

This corpus gives the model a different byte-level distribution: more dialogue, more quotation marks, more character-driven repetition, and more local scene-level variation. These traits make it useful for comparing whether normalization behavior changes when the corpus is shorter, more conversational, and less structurally uniform.

### 7.3 The Adventures of Sherlock Holmes

*The Adventures of Sherlock Holmes* is reserved as an additional corpus section. The uploaded source text is a Project Gutenberg edition of Arthur Conan Doyle’s short-story collection. The file includes front matter, a contents list, story divisions, Roman-numeral section markers, narrative prose, dialogue, mystery-case exposition, proper nouns, dates, and recurring Holmes/Watson conversational structure. :contentReference[oaicite:2]{index=2}

| Property | Value |
|---|---:|
| Total bytes / tokens | pending |
| Training bytes / tokens | pending |
| Validation bytes / tokens | pending |
| Completed RMSNorm runs | 0 |
| Completed MeanAbsNorm runs | 0 |
| Completed MeanAbsNorm-NoCorr runs | 0 |
| Total completed runs | 0 |

Corpus characteristics relevant to byte-level transformer training:

- short-story collection rather than a single continuous argument or single continuous coming-of-age narrative
- recurring Holmes/Watson dialogue structure
- frequent proper nouns, places, dates, clues, and case-specific vocabulary
- strong local story structure with repeated investigative patterning
- mixture of narration, quoted speech, deduction sequences, and exposition
- likely more episodic than *Anne of Green Gables* or *The City of God*

## 8. Metrics

This study reports both raw-summary metrics and curve-derived metrics. Raw-summary metrics come from the per-run result rows written by the training script. Curve-derived metrics come from the per-seed learning-curve CSV files generated during checkpoint evaluation.

The following naming convention is used throughout the report:

| Report Name | Source Field / Computation | Meaning |
|---|---|---|
| Best Val | `best_val_loss` from raw result rows | Lowest validation loss observed during measured curve checkpoints |
| Final Report Val | `val_loss` from raw result rows | Separate final validation evaluation after the training loop |
| Final Curve Val | final `val_loss` entry from the curve CSV | Validation loss at the last measured curve checkpoint |
| Avg Checkpoint Val | mean of `val_loss` across the curve CSV | Average validation behavior across measured checkpoints |
| Curve Overfit Gap | `final_curve_val - best_curve_val` | Amount validation loss rose from the best curve checkpoint to the final curve checkpoint |

Validation loss is computed on randomly sampled validation batches, not by exhaustively evaluating the entire validation split. Each evaluation uses the configured number of validation batches.

### 8.1 Best Validation Loss

Best validation loss is the lowest validation loss observed across measured curve checkpoints in a training run:

\[
L_{\text{best}} = \min_{t \in \mathcal{T}} L_{\text{val}}(t)
\]

where \(\mathcal{T}\) is the set of measured evaluation checkpoints.

In the raw result rows, this metric is stored as:
```txt
best_val_loss
```
This value is based on the checkpoint evaluations recorded during training, not the separate final validation evaluation performed after the training loop.

### 8.2 Final Reported Validation Loss

Final reported validation loss is the separate validation evaluation performed after the training loop finishes:

[
L_{\text{report-final}} = L_{\text{val}}^{\text{post-run}}
]

In the raw result rows, this metric is stored as:

```txt
val_loss
```

This value may differ from the final curve validation loss because it is produced by a separate validation call after training. Since validation batches are sampled from the validation split, the final reported value and the final curve value are not guaranteed to be identical.

### 8.3 Final Curve Validation Loss

Final curve validation loss is the validation loss recorded at the last measured checkpoint in the learning-curve CSV:

[
L_{\text{curve-final}} = L_{\text{val}}(T)
]

where (T) is the final checkpoint step recorded in the curve file.

This metric is used for curve-based comparisons such as average checkpoint validation and curve overfitting gap.

### 8.4 Average Validation Loss Across Checkpoints

Average validation loss across checkpoints is the mean of all validation-loss values recorded in the learning-curve CSV:

[
L_{\text{avg-checkpoint}} = \frac{1}{N}\sum_{t \in \mathcal{T}}L_{\text{val}}(t)
]

where (N) is the number of measured checkpoints.

This metric summarizes validation behavior across the training trajectory instead of only reporting the best or final point.

### 8.5 Curve Overfitting Gap

Curve overfitting gap measures how much validation loss increased after the best measured curve checkpoint:

[
G_{\text{curve-overfit}} = L_{\text{curve-final}} - L_{\text{best}}
]

A larger positive value indicates that validation loss rose more between the best observed checkpoint and the final measured curve checkpoint.

This metric is computed from the curve CSV, not directly from the raw-summary `val_loss` field.

### 8.6 Training Loss

Training loss is the final per-step training loss recorded at the end of each run:

[
L_{\text{train-final}} = L_{\text{train}}(T)
]

In the raw result rows, this metric is stored as:

```txt
train_loss
```

The curve CSV also records `train_window_loss`, which is the average training loss over the configured recent training window at each measured checkpoint. When analyzing validation curves, `train_window_loss` is used to compare training trajectory against checkpoint validation behavior.

### 8.7 Throughput

Throughput is reported in tokens per second.

Training throughput is stored in the raw result rows as:

```txt
train_tok_s
```

Evaluation throughput is stored as:

```txt
eval_tok_s
```

The report may also include step-based and sequence-based timing fields when useful:

```txt
train_steps_s
train_ms_step
train_seq_s
train_us_tok
eval_steps_s
eval_ms_step
eval_seq_s
eval_us_tok
```

### 8.8 Memory

GPU memory is reported in megabytes.

The primary memory fields are:

```txt
peak_alloc_mb
peak_reserved_mb
```

`peak_alloc_mb` reports peak allocated CUDA memory during the measured run. `peak_reserved_mb` reports peak reserved CUDA memory during the measured run.

CUDA peak-memory statistics are reset after warmup, so the reported memory values correspond to the measured training run rather than initialization or warmup artifacts.

## 9. Statistical Analysis

Each normalization method is evaluated across 30 random seeds per completed corpus. Statistical summaries are computed separately for each corpus and normalization method.

The report distinguishes between two levels of aggregation:

1. **Per-run metrics**, where each seed produces one result row for a given normalization method.
2. **Aggregate metrics**, where the 30 per-seed results are summarized by normalization method within each corpus.

The training script writes per-run result rows containing fields such as `train_loss`, `val_loss`, `best_val_loss`, `best_step`, `grad_norm`, `train_tok_s`, `eval_tok_s`, `peak_alloc_mb`, and `peak_reserved_mb`. It then groups results by normalization method and writes aggregate summary fields for each metric. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

For each reported metric, the statistical summary includes:

- mean
- standard deviation
- median
- minimum
- maximum
- 95% confidence interval
- relative percentage difference against RMSNorm

For paired comparisons, the same seed index is compared across normalization methods within the same corpus.

### 9.1 Aggregation Across Seeds

For each corpus, results are grouped by normalization method:

\[
G_{c,m} = \{x_{1}, x_{2}, ..., x_{n}\}
\]

where \(c\) is the corpus, \(m\) is the normalization method, and \(n=30\) for each completed method/corpus pair.

The mean for a metric is:

\[
\bar{x}_{c,m} = \frac{1}{n}\sum_{i=1}^{n}x_i
\]

The median, minimum, and maximum are also reported for each metric:

\[
\operatorname{median}(G_{c,m}), \quad \min(G_{c,m}), \quad \max(G_{c,m})
\]

The existing training script writes mean, standard deviation, minimum, and maximum into the generated summary CSV. Median and confidence intervals are computed from the raw per-seed rows for the manuscript tables.

### 9.2 Confidence Intervals

For each metric, a 95% confidence interval is computed over the 30 seed results:

\[
CI_{95} = \bar{x} \pm t_{0.975,n-1}\frac{s}{\sqrt{n}}
\]

where:

\[
s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}
\]

and \(t_{0.975,n-1}\) is the two-sided critical value from the Student \(t\)-distribution with \(n-1\) degrees of freedom.

For completed corpus/method groups:

\[
n = 30
\]

so the confidence interval uses:

\[
df = 29
\]

### 9.3 Paired Seed Comparisons

Because each normalization method uses the same seed list within a corpus, paired comparisons are computed by matching runs with the same seed.

For a metric \(x\), the paired difference between a tested variant and RMSNorm is:

\[
d_i = x_{i,\text{variant}} - x_{i,\text{RMS}}
\]

The mean paired difference is:

\[
\bar{d} = \frac{1}{n}\sum_{i=1}^{n}d_i
\]

For loss metrics, a negative paired difference means the tested variant has a lower value than RMSNorm for that metric. For throughput metrics, a positive paired difference means the tested variant has higher throughput than RMSNorm. For memory metrics, a negative paired difference means the tested variant uses less memory than RMSNorm.

Paired comparisons are reported separately for:

- MeanAbsNorm vs RMSNorm
- MeanAbsNorm-NoCorr vs RMSNorm
- MeanAbsNorm-NoCorr vs MeanAbsNorm

### 9.4 Effect Sizes

Effect sizes are reported for paired comparisons using the standardized paired mean difference:

\[
d_z = \frac{\bar{d}}{s_d}
\]

where \(s_d\) is the standard deviation of the paired differences:

\[
s_d = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(d_i-\bar{d})^2}
\]

This gives the magnitude of the difference relative to seed-to-seed variability.

Effect sizes are reported alongside raw mean differences because some metrics may show small percentage differences but highly consistent paired behavior across seeds.

### 9.5 Win Counts

Win counts report how often one method has the preferred value across matched seeds.

For loss and memory metrics, a lower value is counted as a win:

\[
\operatorname{win}_i =
\begin{cases}
1, & x_{i,\text{variant}} < x_{i,\text{baseline}} \\
0, & \text{otherwise}
\end{cases}
\]

For throughput metrics, a higher value is counted as a win:

\[
\operatorname{win}_i =
\begin{cases}
1, & x_{i,\text{variant}} > x_{i,\text{baseline}} \\
0, & \text{otherwise}
\end{cases}
\]

The win count is:

\[
W = \sum_{i=1}^{n}\operatorname{win}_i
\]

Win counts are reported as:

\[
W / n
\]

For completed corpus/method comparisons, this is reported out of 30 matched seeds.

### 9.6 Relative Percentage Difference

Relative percentage difference against RMSNorm is reported as:

\[
\Delta_{\%} = 100 \cdot \frac{x_{\text{variant}} - x_{\text{RMS}}}{x_{\text{RMS}}}
\]

For loss and memory metrics, negative values indicate lower values than RMSNorm. For throughput metrics, positive values indicate higher throughput than RMSNorm.

Relative differences are computed using aggregate means unless the table explicitly states that it is using paired per-seed differences.

## 10. Results

### 10.1 Current Data Coverage

The current result set contains completed runs for two corpora: *The City of God* and *Anne of Green Gables*. The Sherlock Holmes corpus is reserved for the next completed result set.

| Corpus | Status | RMSNorm Runs | MeanAbsNorm Runs | MeanAbsNorm-NoCorr Runs | Total Runs |
|---|---:|---:|---:|---:|---:|
| *The City of God* | Complete | 30 | 30 | 30 | 90 |
| *Anne of Green Gables* | Complete | 30 | 30 | 30 | 90 |
| *The Adventures of Sherlock Holmes* | Pending | 0 | 0 | 0 | 0 |

The completed result set currently contains 180 total runs.

### 10.2 Overall Summary

The table below reports aggregate mean values for each completed corpus and normalization method.

| Corpus      | Norm        | Best Val | Final Report Val | Avg Checkpoint Val | Curve Overfit Gap | Train tok/s | Eval tok/s | Reserved MB |
| ----------- | ----------- | -------: | ---------------: | -----------------: | ----------------: | ----------: | ---------: | ----------: |
| City of God | RMSNorm     |   1.3984 |           1.5323 |             1.7669 |            0.1572 |     15889.8 |    45455.7 |       10788 |
| City of God | MeanAbsNorm |   1.4022 |           1.5059 |             1.7752 |            0.1264 |     15690.4 |    44882.1 |       10436 |
| City of God | NoCorr      |   1.6602 |           1.6469 |             2.1897 |            0.0026 |     15928.6 |    45520.6 |       10436 |
| Anne        | RMSNorm     |   1.6581 |           3.1545 |             2.3210 |            1.5167 |     15697.9 |    44722.4 |       10788 |
| Anne        | MeanAbsNorm |   1.6553 |           3.0839 |             2.2831 |            1.4445 |     15874.8 |    45342.9 |       10436 |
| Anne        | NoCorr      |   1.6554 |           2.3539 |             2.0424 |            0.7153 |     15601.4 |    44620.1 |       10436 |

Across the completed corpora, the normalization methods show different behavior depending on corpus material. On *The City of God*, RMSNorm has the lowest best validation loss and lowest average checkpoint validation loss, while MeanAbsNorm has the lowest final reported validation loss and a smaller curve overfitting gap than RMSNorm. On *Anne of Green Gables*, MeanAbsNorm has slightly lower best validation loss, final reported validation loss, average checkpoint validation loss, curve overfitting gap, higher throughput, and lower reserved memory than RMSNorm. MeanAbsNorm-NoCorr shows the largest corpus-dependent shift, with weak validation behavior on *The City of God* but strong late-curve validation behavior on *Anne of Green Gables*.

### 10.3 Aggregate Tables

This section reports expanded aggregate statistics for each corpus and normalization method, including mean, standard deviation, median, minimum, maximum, and 95% confidence intervals.

#### 10.3.1 The City of God Aggregate Statistics

| Metric | RMSNorm Mean | RMSNorm Std | MeanAbsNorm Mean | MeanAbsNorm Std | NoCorr Mean | NoCorr Std |
|---|---:|---:|---:|---:|---:|---:|
| Best Val | | | | | | |
| Final Report Val | | | | | | |
| Avg Checkpoint Val | | | | | | |
| Curve Overfit Gap | | | | | | |
| Train tok/s | | | | | | |
| Eval tok/s | | | | | | |
| Reserved MB | | | | | | |

#### 10.3.2 Anne of Green Gables Aggregate Statistics

| Metric | RMSNorm Mean | RMSNorm Std | MeanAbsNorm Mean | MeanAbsNorm Std | NoCorr Mean | NoCorr Std |
|---|---:|---:|---:|---:|---:|---:|
| Best Val | | | | | | |
| Final Report Val | | | | | | |
| Avg Checkpoint Val | | | | | | |
| Curve Overfit Gap | | | | | | |
| Train tok/s | | | | | | |
| Eval tok/s | | | | | | |
| Reserved MB | | | | | | |

### 10.4 Relative Differences vs RMSNorm

Relative differences are computed against RMSNorm within the same corpus. For loss and memory metrics, negative values indicate lower values than RMSNorm. For throughput metrics, positive values indicate higher throughput than RMSNorm.

\[
\Delta_{\%} = 100 \cdot \frac{x_{\text{variant}} - x_{\text{RMS}}}{x_{\text{RMS}}}
\]

#### 10.4.1 The City of God Relative Differences

| Norm | Best Val Δ% | Final Report Val Δ% | Avg Checkpoint Val Δ% | Curve Overfit Gap Δ% | Train tok/s Δ% | Eval tok/s Δ% | Reserved MB Δ% |
|---|---:|---:|---:|---:|---:|---:|---:|
| MeanAbsNorm | +0.27% | -1.72% | +0.47% | -19.59% | -1.25% | -1.26% | -3.26% |
| NoCorr | +18.72% | +7.48% | +23.93% | -98.35% | +0.24% | +0.14% | -3.26% |

#### 10.4.2 Anne of Green Gables Relative Differences

| Norm | Best Val Δ% | Final Report Val Δ% | Avg Checkpoint Val Δ% | Curve Overfit Gap Δ% | Train tok/s Δ% | Eval tok/s Δ% | Reserved MB Δ% |
|---|---:|---:|---:|---:|---:|---:|---:|
| MeanAbsNorm | -0.17% | -2.24% | -1.63% | -4.76% | +1.13% | +1.39% | -3.26% |
| NoCorr | -0.16% | -25.38% | -12.00% | -52.84% | -0.61% | -0.23% | -3.26% |

### 10.5 Curves and Figures

The curve figures plot training-window loss, validation loss, best validation loss so far, and generalization gap across measured checkpoints. These figures are used to inspect validation trajectory shape, overfitting onset, and differences between best, average, and final curve behavior.

Recommended figures for this section:

| Figure | Description |
|---|---|
| Figure 1 | *The City of God*: validation curves by normalization method |
| Figure 2 | *Anne of Green Gables*: validation curves by normalization method |
| Figure 3 | Best validation loss by corpus and normalization method |
| Figure 4 | Final reported validation loss by corpus and normalization method |
| Figure 5 | Average checkpoint validation loss by corpus and normalization method |
| Figure 6 | Curve overfitting gap by corpus and normalization method |
| Figure 7 | Training throughput by corpus and normalization method |
| Figure 8 | Reserved GPU memory by corpus and normalization method |

Expecting further corpora data

## 11. Per-Corpus Analysis

### 11.1 The City of God

*The City of God* shows the clearest separation between RMSNorm, corrected MeanAbsNorm, and MeanAbsNorm-NoCorr.

| Norm | Best Val | Final Report Val | Avg Checkpoint Val | Curve Overfit Gap | Train tok/s | Eval tok/s | Reserved MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| RMSNorm | 1.3984 | 1.5323 | 1.7669 | 0.1572 | 15889.8 | 45455.7 | 10788 |
| MeanAbsNorm | 1.4022 | 1.5059 | 1.7752 | 0.1264 | 15690.4 | 44882.1 | 10436 |
| NoCorr | 1.6602 | 1.6469 | 2.1897 | 0.0026 | 15928.6 | 45520.6 | 10436 |

RMSNorm reaches the lowest best validation loss on this corpus, with a mean best validation loss of `1.3984` compared with `1.4022` for MeanAbsNorm and `1.6602` for MeanAbsNorm-NoCorr. RMSNorm also has the lowest average checkpoint validation loss at `1.7669`, compared with `1.7752` for MeanAbsNorm and `2.1897` for NoCorr.

MeanAbsNorm has lower final reported validation loss than RMSNorm, with `1.5059` compared with `1.5323`. It also has a smaller curve overfitting gap, `0.1264` compared with `0.1572`, suggesting that its late-curve validation rise is smaller under this corpus/configuration. However, MeanAbsNorm does not improve the best validation point or the mean validation behavior across checkpoints on this corpus.

MeanAbsNorm-NoCorr behaves differently from both corrected MeanAbsNorm and RMSNorm. Its best validation loss, final reported validation loss, and average checkpoint validation loss are all higher than the corrected and RMS variants. Its very small curve overfitting gap, `0.0026`, should be interpreted together with its weaker validation trajectory: the small gap does not indicate stronger overall validation behavior, because the method does not reach a comparably low best validation loss in the first place.

Throughput and memory show a different tradeoff pattern. NoCorr has the highest train and eval throughput on this corpus, while RMSNorm is close behind. MeanAbsNorm is slightly slower than RMSNorm. Both MeanAbsNorm and NoCorr use lower reserved GPU memory, `10436 MB`, compared with `10788 MB` for RMSNorm.

Overall, *The City of God* favors RMSNorm on best and average validation behavior, while MeanAbsNorm shows better final reported validation and a smaller late-curve overfitting gap. NoCorr is computationally competitive but weaker on the main validation-loss metrics.

### 11.2 Anne of Green Gables

*Anne of Green Gables* shows a different pattern from *The City of God*. The corrected MeanAbsNorm variant is slightly lower than RMSNorm across all reported validation and runtime summary metrics in the current aggregate table, while NoCorr shows unusually strong late-validation behavior.

| Norm | Best Val | Final Report Val | Avg Checkpoint Val | Curve Overfit Gap | Train tok/s | Eval tok/s | Reserved MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| RMSNorm | 1.6581 | 3.1545 | 2.3210 | 1.5167 | 15697.9 | 44722.4 | 10788 |
| MeanAbsNorm | 1.6553 | 3.0839 | 2.2831 | 1.4445 | 15874.8 | 45342.9 | 10436 |
| NoCorr | 1.6554 | 2.3539 | 2.0424 | 0.7153 | 15601.4 | 44620.1 | 10436 |

MeanAbsNorm has the lowest best validation loss among the corrected/RMS comparison, with `1.6553` compared with `1.6581` for RMSNorm. The difference is small, but it is directionally consistent with the final reported validation loss and average checkpoint validation loss: MeanAbsNorm reports `3.0839` final validation compared with `3.1545` for RMSNorm, and `2.2831` average checkpoint validation compared with `2.3210`.

MeanAbsNorm also has a smaller curve overfitting gap than RMSNorm, `1.4445` compared with `1.5167`. This indicates that both methods overfit substantially on this corpus by the end of the run, but the corrected mean-absolute variant rises slightly less from its best curve point to its final curve point.

NoCorr is the most unusual result on this corpus. Its best validation loss, `1.6554`, is nearly identical to MeanAbsNorm, and its final reported validation loss, average checkpoint validation loss, and curve overfitting gap are all substantially lower than RMSNorm and corrected MeanAbsNorm. However, this behavior should be interpreted alongside its weaker training-loss and gradient-norm behavior in the raw summaries. The validation curve behavior is strong, but the optimization behavior is not simply equivalent to the corrected variant.

For runtime characteristics, MeanAbsNorm has the highest train and eval throughput on this corpus, while RMSNorm is second and NoCorr is slightly slower. Both MeanAbsNorm and NoCorr use lower reserved GPU memory than RMSNorm, with `10436 MB` compared with `10788 MB`.

Overall, *Anne of Green Gables* shows stronger results for mean-absolute scaling than *The City of God*. Corrected MeanAbsNorm improves over RMSNorm across the reported aggregate metrics, while NoCorr shows the strongest late-validation behavior but remains the most methodologically unusual variant because its validation improvement is paired with less favorable optimization signals.

### 11.3 Sherlock Holmes

The Sherlock Holmes corpus section is reserved for the next completed result set. No completed aggregate table is currently available for this corpus.

| Norm | Status |
|---|---|
| RMSNorm | Pending |
| MeanAbsNorm | Pending |
| MeanAbsNorm-NoCorr | Pending |

This corpus should be analyzed using the same structure as the completed corpora:

- best validation loss
- final reported validation loss
- average checkpoint validation loss
- curve overfitting gap
- training loss
- throughput
- reserved GPU memory
- curve shape
- comparison against RMSNorm
- corrected MeanAbsNorm vs NoCorr behavior

Because Sherlock Holmes is structurally different from both completed corpora, it should be useful for testing whether the normalization patterns observed so far persist on a shorter, episodic mystery-fiction corpus. The expected analysis focus should be whether its story-by-story structure, recurring Holmes/Watson dialogue, and repeated investigative patterning produce validation behavior closer to *Anne of Green Gables* or closer to *The City of God*.

## 12. Cross-Corpus Analysis

The completed corpora show that normalization behavior changes with corpus material. The same model configuration, tokenization method, seed count, split strategy, and training schedule produce different validation and runtime patterns on *The City of God* and *Anne of Green Gables*.

### 12.1 Validation-Loss Trends

Validation-loss behavior differs substantially between the two completed corpora.

On *The City of God*, RMSNorm has the lowest best validation loss and lowest average checkpoint validation loss:

| Metric | RMSNorm | MeanAbsNorm | NoCorr |
|---|---:|---:|---:|
| Best Val | 1.3984 | 1.4022 | 1.6602 |
| Avg Checkpoint Val | 1.7669 | 1.7752 | 2.1897 |

MeanAbsNorm has lower final reported validation loss than RMSNorm:

| Metric | RMSNorm | MeanAbsNorm | NoCorr |
|---|---:|---:|---:|
| Final Report Val | 1.5323 | 1.5059 | 1.6469 |

On *Anne of Green Gables*, MeanAbsNorm improves over RMSNorm across the main validation summary metrics:

| Metric | RMSNorm | MeanAbsNorm | NoCorr |
|---|---:|---:|---:|
| Best Val | 1.6581 | 1.6553 | 1.6554 |
| Final Report Val | 3.1545 | 3.0839 | 2.3539 |
| Avg Checkpoint Val | 2.3210 | 2.2831 | 2.0424 |

NoCorr shows the largest change between corpora. On *The City of God*, NoCorr has much higher best validation loss and average checkpoint validation loss than RMSNorm and corrected MeanAbsNorm. On *Anne of Green Gables*, NoCorr has the lowest final reported validation loss and lowest average checkpoint validation loss, despite not showing the same pattern on *The City of God*.

### 12.2 Overfitting Behavior

Curve overfitting gap is computed as:

\[
G_{\text{curve-overfit}} = L_{\text{curve-final}} - L_{\text{best}}
\]

On *The City of God*, MeanAbsNorm has a smaller curve overfitting gap than RMSNorm:

| Corpus | RMSNorm | MeanAbsNorm | NoCorr |
|---|---:|---:|---:|
| City of God | 0.1572 | 0.1264 | 0.0026 |

The NoCorr gap is very small on this corpus, but it should be interpreted alongside its much weaker validation-loss levels. A small overfitting gap is less meaningful when the method does not reach a comparably low best validation loss.

On *Anne of Green Gables*, all methods show larger late-curve validation rise than on *The City of God*, but the magnitude differs:

| Corpus | RMSNorm | MeanAbsNorm | NoCorr |
|---|---:|---:|---:|
| Anne of Green Gables | 1.5167 | 1.4445 | 0.7153 |

MeanAbsNorm has a smaller overfitting gap than RMSNorm on both completed corpora. NoCorr has the smallest overfitting gap on both corpora, but its meaning differs by corpus: on *The City of God*, it coincides with weak validation performance; on *Anne of Green Gables*, it coincides with strong late-validation behavior.

### 12.3 Corpus Sensitivity

The main cross-corpus finding is that normalization behavior is corpus-sensitive.

*The City of God* is larger, more formal, more repetitive, and more argumentative in structure. Under this corpus condition, RMSNorm has stronger best and average validation behavior, while MeanAbsNorm shows better final reported validation and a smaller overfitting gap.

*Anne of Green Gables* is shorter, more narrative, more dialogue-heavy, and more locally varied. Under this corpus condition, MeanAbsNorm improves over RMSNorm across best validation loss, final reported validation loss, average checkpoint validation loss, overfitting gap, throughput, and reserved memory.

NoCorr is the strongest example of corpus sensitivity. It performs poorly on *The City of God* by best and average validation loss, but shows unexpectedly strong late-validation behavior on *Anne of Green Gables*. This suggests that uncorrected mean-absolute scaling may interact more sharply with corpus structure than either RMSNorm or corrected MeanAbsNorm.

### 12.4 Speed and Memory Tradeoffs

Runtime and memory results do not follow exactly the same pattern as validation loss.

On *The City of God*, NoCorr has the highest throughput, while MeanAbsNorm is slightly slower than RMSNorm:

| Norm | Train tok/s | Eval tok/s | Reserved MB |
|---|---:|---:|---:|
| RMSNorm | 15889.8 | 45455.7 | 10788 |
| MeanAbsNorm | 15690.4 | 44882.1 | 10436 |
| NoCorr | 15928.6 | 45520.6 | 10436 |

On *Anne of Green Gables*, MeanAbsNorm has the highest throughput, while NoCorr is slightly slower than RMSNorm:

| Norm | Train tok/s | Eval tok/s | Reserved MB |
|---|---:|---:|---:|
| RMSNorm | 15697.9 | 44722.4 | 10788 |
| MeanAbsNorm | 15874.8 | 45342.9 | 10436 |
| NoCorr | 15601.4 | 44620.1 | 10436 |

Both MeanAbsNorm and NoCorr use lower reserved GPU memory than RMSNorm on both completed corpora:

\[
10436\text{ MB} \quad \text{vs} \quad 10788\text{ MB}
\]

This is a consistent reserved-memory reduction of approximately:

\[
-3.26\%
\]

Throughput is less consistent than memory. MeanAbsNorm is slower than RMSNorm on *The City of God* but faster on *Anne of Green Gables*. NoCorr is slightly faster than RMSNorm on *The City of God* but slightly slower on *Anne of Green Gables*.

### 12.5 NoCorr Behavior

MeanAbsNorm-NoCorr is the least stable variant across corpora.

On *The City of God*, NoCorr has substantially worse validation metrics:

| Metric | RMSNorm | MeanAbsNorm | NoCorr |
|---|---:|---:|---:|
| Best Val | 1.3984 | 1.4022 | 1.6602 |
| Final Report Val | 1.5323 | 1.5059 | 1.6469 |
| Avg Checkpoint Val | 1.7669 | 1.7752 | 2.1897 |

On *Anne of Green Gables*, NoCorr has strong late-validation metrics:

| Metric | RMSNorm | MeanAbsNorm | NoCorr |
|---|---:|---:|---:|
| Best Val | 1.6581 | 1.6553 | 1.6554 |
| Final Report Val | 3.1545 | 3.0839 | 2.3539 |
| Avg Checkpoint Val | 2.3210 | 2.2831 | 2.0424 |
| Curve Overfit Gap | 1.5167 | 1.4445 | 0.7153 |

This makes NoCorr an important ablation rather than a disposable control. Removing the correction constant does not merely produce a uniformly worse version of MeanAbsNorm. Instead, it changes the training and validation behavior in a corpus-dependent way.

The current data supports three observations:

- NoCorr is much weaker than RMSNorm and corrected MeanAbsNorm on *The City of God* validation metrics.
- NoCorr shows unusually strong final and average validation behavior on *Anne of Green Gables*.
- NoCorr’s late-validation behavior should be interpreted alongside training loss, gradient norm, and curve shape rather than treated as a standalone improvement.

The NoCorr results justify keeping the correction-constant ablation in future runs, especially on Sherlock Holmes and any additional corpora.

## 13. Discussion

The completed results show that the normalization scale statistic changes transformer training behavior even when architecture, optimizer, tokenizer, seed count, training duration, evaluation interval, and train/validation split are held constant.

The strongest pattern across the completed corpora is corpus sensitivity. On *The City of God*, RMSNorm has the lowest best validation loss and lowest average checkpoint validation loss, while MeanAbsNorm has lower final reported validation loss and a smaller curve overfitting gap. On *Anne of Green Gables*, MeanAbsNorm is lower than RMSNorm across best validation loss, final reported validation loss, average checkpoint validation loss, curve overfitting gap, throughput, and reserved memory.

This indicates that corrected mean-absolute scaling is not behaving as a trivial replacement of RMS scaling. It produces a different validation trajectory, and the direction of that difference depends on the corpus. The difference is especially visible when comparing best validation loss against final reported validation loss and average checkpoint validation loss. A method can reach a strong best checkpoint while still showing a sharper late-curve rise, so the report tracks multiple validation metrics instead of reducing the comparison to one number.

The correction constant also matters. MeanAbsNorm-NoCorr does not behave like a uniformly weaker copy of corrected MeanAbsNorm. On *The City of God*, NoCorr has much worse best and average validation loss than RMSNorm and corrected MeanAbsNorm. On *Anne of Green Gables*, NoCorr has much stronger final reported validation loss, average checkpoint validation loss, and curve overfitting gap than either corrected MeanAbsNorm or RMSNorm. This makes NoCorr an important ablation because removing the correction constant changes the optimization and validation behavior in a corpus-dependent way.

Runtime and memory show a separate pattern from validation loss. MeanAbsNorm and NoCorr both use less reserved GPU memory than RMSNorm in the completed results. Throughput is less consistent: MeanAbsNorm is slightly slower than RMSNorm on *The City of God* but faster on *Anne of Green Gables*. This separates the practical runtime tradeoff from the validation-loss tradeoff.

The current evidence supports three main observations:

1. RMS-based, corrected mean-absolute, and uncorrected mean-absolute scaling produce measurably different training behavior in the same transformer architecture.
2. Corpus structure affects the observed validation and overfitting patterns.
3. MeanAbsNorm-NoCorr is the most corpus-sensitive variant in the completed results.

The next important comparison is Sherlock Holmes, because it differs from both completed corpora. It is more episodic than *Anne of Green Gables* and less formally argumentative than *The City of God*. Its results should help clarify whether the Anne pattern is specific to that text or more generally associated with narrative/dialogue-heavy corpora.

## 14. Threats to Validity

### 14.1 Model Scale

The current results are based on one decoder-only transformer configuration. The observed normalization behavior may change with larger or smaller models, different layer counts, wider hidden dimensions, different attention-head counts, or different feedforward ratios.

### 14.2 Corpus Selection

The completed corpora are individual literary texts rather than broad mixed datasets. Results from *The City of God* and *Anne of Green Gables* may not represent behavior on code, scientific text, web text, dialogue datasets, multilingual text, or benchmark language-modeling corpora.

### 14.3 Sequential Train/Validation Split

Each corpus is split sequentially, with the first 90% used for training and the final 10% used for validation. This measures generalization to the later portion of the same text, not performance on a randomly shuffled validation subset. This split is useful for studying within-text continuation behavior, but it can interact with narrative structure, chapter placement, topic shifts, or end-of-book distribution changes.

### 14.4 Byte-Level Tokenization

The study uses byte-level tokenization with a vocabulary size of 256. Byte-level training exposes raw punctuation, capitalization, whitespace, encoding artifacts, and Project Gutenberg formatting directly to the model. Results may differ under BPE, unigram, wordpiece, or other subword tokenizers.

### 14.5 Validation Sampling

Validation loss is estimated from sampled validation batches rather than an exhaustive pass over the entire validation split. Final reported validation loss and final curve validation loss can differ because they are produced by separate sampled validation calls.

### 14.6 Training Duration

The completed runs use 5000 measured training steps. Longer runs may change the relative behavior of the methods, especially for overfitting dynamics and NoCorr behavior. Shorter runs may emphasize early convergence, while longer runs may emphasize late generalization stability.

### 14.7 Hyperparameter Specificity

The results depend on the selected learning rate, batch size, sequence length, optimizer, gradient clipping value, evaluation interval, and warmup duration. Different hyperparameter choices may change the relative behavior of the normalization methods.

### 14.8 Implementation Specificity

The tested methods use a specific implementation of RMSNorm, corrected MeanAbsNorm, and MeanAbsNorm-NoCorr. Results may change with different epsilon placement, fused kernels, alternative numerical precision, different compilation settings, or optimized CUDA/Triton implementations.

### 14.9 Corpus Metadata and Formatting

The source texts include Project Gutenberg front matter and formatting artifacts. Because the tokenizer is byte-level, these artifacts are part of the training distribution unless explicitly removed. This may affect validation behavior, especially when the train/validation split crosses structural boundaries in the text.

## 15. Reproducibility

### 15.1 Repository Structure

The repository stores aggregate result files, per-seed raw results, per-seed curve CSVs, and curve plots by corpus and normalization method.

```txt
.
├── README.md
├── LICENSE
├── The_City_of_God/
│   ├── norm_runs_raw_20260422_033233.csv
│   ├── norm_runs_summary_20260422_033233.csv
│   ├── norm_runs_report_20260422_033233.txt
│   ├── norm_runs_meta_20260422_033233.json
│   ├── RMSNorm/
│   ├── MANorm/
│   └── NoCorrNorm/
├── Anne_of_Green_Gables/
│   ├── norm_runs_raw_20260423_210449.csv
│   ├── norm_runs_summary_20260423_210449.csv
│   ├── norm_runs_report_20260423_210449.txt
│   ├── norm_runs_meta_20260423_210449.json
│   ├── RMSNorm/
│   ├── MANorm/
│   └── NoCorrNorm/
└── The_Adventures_of_Sherlock_Holmes/
```
Each normalization-method folder contains per-seed curve CSV files and corresponding curve plots.

### 15.2 Exact Commands

The completed runs use the same architecture and training configuration across normalization methods. The command structure is:
```bash
python norm_ablation_multiseed_clean_progress_fixed.py \
  --data <corpus-file>.txt \
  --norms rms,meanabs,meanabs_nocorr \
  --steps 5000 \
  --warmup-steps 50 \
  --eval-batches 32 \
  --eval-every 500 \
  --train-window 100 \
  --seq-len 1024 \
  --batch-size 4 \
  --d-model 704 \
  --n-heads 22 \
  --n-layers 15 \
  --ffn-hidden 2816 \
  --lr 3e-4 \
  --eps 1e-5 \
  --grad-clip 1.0 \
  --compile-backend none \
  --seeds "598422775,1786548453,1426323417,736896387,1722602715,1639699922,1048955721,370076197,188292180,482242919,676750256,405379411,130604721,1310941852,1162914961,1109572572,1431649702,1497939373,741210827,129344154,273318888,1718616343,1970508508,336498026,1301334618,1258359901,2022903755,281919623,922641176,1225032379" \
  --outdir <output-directory> \
  --tag <run-tag>
```
City of God run:
```bash
python norm_ablation_multiseed_clean_progress_fixed.py \
  --data TheCityOfGod.txt \
  --norms rms,meanabs,meanabs_nocorr \
  --steps 5000 \
  --warmup-steps 50 \
  --eval-batches 32 \
  --eval-every 500 \
  --train-window 100 \
  --seq-len 1024 \
  --batch-size 4 \
  --d-model 704 \
  --n-heads 22 \
  --n-layers 15 \
  --ffn-hidden 2816 \
  --lr 3e-4 \
  --eps 1e-5 \
  --grad-clip 1.0 \
  --compile-backend none \
  --seeds "598422775,1786548453,1426323417,736896387,1722602715,1639699922,1048955721,370076197,188292180,482242919,676750256,405379411,130604721,1310941852,1162914961,1109572572,1431649702,1497939373,741210827,129344154,273318888,1718616343,1970508508,336498026,1301334618,1258359901,2022903755,281919623,922641176,1225032379" \
  --outdir The_City_of_God \
  --tag 20260422_033233
```
Anne of Green Gables run:
```bash
python norm_ablation_multiseed_clean_progress_fixed.py \
  --data Anne_of_Green_Gables.txt \
  --norms rms,meanabs,meanabs_nocorr \
  --steps 5000 \
  --warmup-steps 50 \
  --eval-batches 32 \
  --eval-every 500 \
  --train-window 100 \
  --seq-len 1024 \
  --batch-size 4 \
  --d-model 704 \
  --n-heads 22 \
  --n-layers 15 \
  --ffn-hidden 2816 \
  --lr 3e-4 \
  --eps 1e-5 \
  --grad-clip 1.0 \
  --compile-backend none \
  --seeds "598422775,1786548453,1426323417,736896387,1722602715,1639699922,1048955721,370076197,188292180,482242919,676750256,405379411,130604721,1310941852,1162914961,1109572572,1431649702,1497939373,741210827,129344154,273318888,1718616343,1970508508,336498026,1301334618,1258359901,2022903755,281919623,922641176,1225032379" \
  --outdir Anne_of_Green_Gables \
  --tag 20260423_210449
```
Sherlock Holmes run:
```bash
python norm_ablation_multiseed_clean_progress_fixed.py \
  --data The_Adventures_of_Sherlock_Holmes.txt \
  --norms rms,meanabs,meanabs_nocorr \
  --steps 5000 \
  --warmup-steps 50 \
  --eval-batches 32 \
  --eval-every 500 \
  --train-window 100 \
  --seq-len 1024 \
  --batch-size 4 \
  --d-model 704 \
  --n-heads 22 \
  --n-layers 15 \
  --ffn-hidden 2816 \
  --lr 3e-4 \
  --eps 1e-5 \
  --grad-clip 1.0 \
  --compile-backend none \
  --seeds "598422775,1786548453,1426323417,736896387,1722602715,1639699922,1048955721,370076197,188292180,482242919,676750256,405379411,130604721,1310941852,1162914961,1109572572,1431649702,1497939373,741210827,129344154,273318888,1718616343,1970508508,336498026,1301334618,1258359901,2022903755,281919623,922641176,1225032379" \
  --outdir The_Adventures_of_Sherlock_Holmes \
  --tag <new-run-tag>
```

### 15.3 Environment

### 15.3 Environment

Experiments were run on a Windows workstation with an NVIDIA GeForce RTX 5080 GPU using CUDA through PyTorch.

| Component | Value |
|---|---|
| Operating system | Windows 11 |
| Python | 3.14.3 |
| PyTorch | 2.11.0+cu130 |
| PyTorch CUDA build | 13.0 |
| CUDA available | True |
| GPU | NVIDIA GeForce RTX 5080 |
| Device backend | CUDA |
| NVIDIA driver | 595.97 |
| Local CUDA toolkit | 12.9 |
| NumPy | 2.4.4 |

### 15.4 Data Processing

Each corpus file is read as raw bytes:
```python
raw = path.read_bytes()
full = torch.tensor(list(raw), dtype=torch.long)
```
The train/validation split is sequential:
```python
split = int(0.9 * full.numel())
train = full[:split]
val = full[split:]
```
Training and validation batches are sampled from their respective byte ranges. For each sampled start position, the input sequence and target sequence are:
```python
x = source[s : s + seq_len]
y = source[s + 1 : s + seq_len + 1]
```
No text cleaning, BPE tokenization, word-level tokenization, sentencepiece processing, lowercasing, punctuation removal, or randomized train/validation split is applied.

### 15.5 Result Files

Each completed corpus includes four main result file types:

| File Type                     | Description                                       |
| ----------------------------- | ------------------------------------------------- |
| `norm_runs_raw_<tag>.csv`     | One row per norm/seed run                         |
| `norm_runs_summary_<tag>.csv` | Aggregate summary grouped by normalization method |
| `norm_runs_report_<tag>.txt`  | Text report generated by the training script      |
| `norm_runs_meta_<tag>.json`   | Run metadata and output-file paths                |

Each normalization-method folder also contains:

| File Type                          | Description                       |
| ---------------------------------- | --------------------------------- |
| `curve_<norm>_seed<seed>.csv`      | Per-seed learning-curve data      |
| `curve_<norm>_seed<seed>_plot.png` | Plot generated from the curve CSV |

The curve CSV files contain checkpoint-level values such as:

- step
- train_window_loss
- val_loss
- generalization_gap
- best_val_so_far
- best_step_so_far

## 16. Future Work

Future work should extend the study across more corpora, model scales, tokenizers, and training durations.

Planned extensions:

- Complete the Sherlock Holmes run using the same configuration and seed list.
- Add the expanded aggregate tables with standard deviation, median, minimum, maximum, confidence intervals, paired seed comparisons, effect sizes, and win counts.
- Add full per-seed statistical appendices.
- Run longer 10k and 20k step experiments to examine late-training behavior.
- Test larger model configurations to see whether the normalization patterns persist with scale.
- Test smaller model configurations to determine whether the observed behavior is size-dependent.
- Compare byte-level tokenization against BPE or another subword tokenizer.
- Add randomized train/validation split experiments alongside the current sequential split.
- Add broader corpus categories, including code, scientific writing, standard modern prose, and dialogue-heavy text.
- Analyze gradient-norm behavior across normalization methods, especially for MeanAbsNorm-NoCorr.
- Compare validation curve shape across corpora using best-step location, overfit onset, and post-best slope.
- Add statistical plots for confidence intervals and paired seed differences.
- Add implementation benchmarks for alternative MeanAbsNorm forms, including reciprocal-based variants and fused implementations.
- Evaluate whether the reserved-memory difference remains consistent under different batch sizes and sequence lengths.
- Add direct LayerNorm comparison runs as a secondary baseline.
- Add metadata checks to prevent path-label leakage across corpus folders.

## References

Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. arXiv:1607.06450. :contentReference[oaicite:0]{index=0}

Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization*. Advances in Neural Information Processing Systems 32. arXiv:1910.07467. :contentReference[oaicite:1]{index=1}

Hoffer, E., Banner, R., Golan, I., & Soudry, D. (2018). *Norm Matters: Efficient and Accurate Normalization Schemes in Deep Networks*. Advances in Neural Information Processing Systems 31. :contentReference[oaicite:2]{index=2}

Montgomery, L. M. *Anne of Green Gables*. Project Gutenberg eBook #45. Source corpus used for the Anne of Green Gables experiments. :contentReference[oaicite:3]{index=3}

Doyle, A. C. *The Adventures of Sherlock Holmes*. Project Gutenberg eBook #1661. Reserved source corpus for the Sherlock Holmes experiments. :contentReference[oaicite:4]{index=4}

Augustine of Hippo. *The City of God, Volume I*. Translated by Marcus Dods. Project Gutenberg eBook #45304. Source corpus used for the City of God experiments. :contentReference[oaicite:5]{index=5}

## Appendix A: Full Raw Result Tables

This appendix contains the full raw per-run result tables for each completed corpus. Each row corresponds to one normalization method and one random seed.

Raw result fields include:

- `seed`
- `norm`
- `params`
- `device`
- `train_bytes`
- `val_bytes`
- `train_loss`
- `val_loss`
- `best_val_loss`
- `best_step`
- `overfit_step`
- `overfit_val_loss`
- `grad_norm`
- `train_elapsed_s`
- `train_steps_s`
- `train_ms_step`
- `train_seq_s`
- `train_tok_s`
- `train_us_tok`
- `eval_elapsed_s`
- `eval_steps_s`
- `eval_ms_step`
- `eval_seq_s`
- `eval_tok_s`
- `eval_us_tok`
- `peak_alloc_mb`
- `peak_reserved_mb`
- `curve_path`

### Appendix A.1 The City of God Raw Results

Source file:

```txt
The_City_of_God/norm_runs_raw_20260422_033233.csv
```
Insert full raw CSV table here or link to the repository file.

### Appendix A.2 Anne of Green Gables Raw Results

Source file:

```txt
Anne_of_Green_Gables/norm_runs_raw_20260423_210449.csv
```
Insert full raw CSV table here or link to the repository file.

### Appendix A.3 Sherlock Holmes Raw Results

Pending.

Source file after completion:

The_Adventures_of_Sherlock_Holmes/norm_runs_raw_<tag>.csv

## Appendix B: Per-Seed Results

This appendix reports per-seed comparisons across normalization methods. Per-seed tables are used to inspect whether aggregate differences are driven by broad consistency across seeds or by a small number of outlier runs.

### Appendix B.1 Per-Seed Best Validation Loss

| Corpus | Seed | RMSNorm | MeanAbsNorm | MeanAbsNorm-NoCorr |
|---|---:|---:|---:|---:|
| City of God | | | | |
| Anne of Green Gables | | | | |

### Appendix B.2 Per-Seed Final Reported Validation Loss

| Corpus | Seed | RMSNorm | MeanAbsNorm | MeanAbsNorm-NoCorr |
|---|---:|---:|---:|---:|
| City of God | | | | |
| Anne of Green Gables | | | | |

### Appendix B.3 Per-Seed Average Checkpoint Validation Loss

| Corpus | Seed | RMSNorm | MeanAbsNorm | MeanAbsNorm-NoCorr |
|---|---:|---:|---:|---:|
| City of God | | | | |
| Anne of Green Gables | | | | |

### Appendix B.4 Per-Seed Curve Overfitting Gap

| Corpus | Seed | RMSNorm | MeanAbsNorm | MeanAbsNorm-NoCorr |
|---|---:|---:|---:|---:|
| City of God | | | | |
| Anne of Green Gables | | | | |

### Appendix B.5 Per-Seed Throughput

| Corpus | Seed | Norm | Train tok/s | Eval tok/s |
|---|---:|---|---:|---:|
| City of God | | | | |
| Anne of Green Gables | | | | |

### Appendix B.6 Per-Seed Memory

| Corpus | Seed | Norm | Peak Alloc MB | Peak Reserved MB |
|---|---:|---|---:|---:|
| City of God | | | | |
| Anne of Green Gables | | | | |

## Appendix C: Additional Plots

This appendix contains plots generated from the per-seed curve CSV files. Each curve CSV records checkpoint-level training-window loss, validation loss, best validation loss so far, and generalization gap.

Curve CSV fields include:

- `seed`
- `norm`
- `step`
- `train_window_loss`
- `val_loss`
- `generalization_gap`
- `best_val_so_far`
- `best_step_so_far`

### Appendix C.1 The City of God Curves

Include or link:

The_City_of_God/RMSNorm/
The_City_of_God/MANorm/
The_City_of_God/NoCorrNorm/
Recommended plots:

RMSNorm validation curves
MeanAbsNorm validation curves
MeanAbsNorm-NoCorr validation curves
combined best-validation comparison
combined final-curve validation comparison
combined generalization-gap comparison

### Appendix C.2 Anne of Green Gables Curves

Include or link:

Anne_of_Green_Gables/RMSNorm/
Anne_of_Green_Gables/MANorm/
Anne_of_Green_Gables/NoCorrNorm/

Recommended plots:

RMSNorm validation curves
MeanAbsNorm validation curves
MeanAbsNorm-NoCorr validation curves
combined best-validation comparison
combined final-curve validation comparison
combined generalization-gap comparison

### Appendix C.3 Sherlock Holmes Curves

Pending.

## Appendix D: Implementation Listings

This appendix contains the exact normalization implementations used in the experiment script. The tested normalization variants are implemented as PyTorch modules and selected through the `make_norm` factory function. The transformer block applies the selected normalization method in pre-norm position before attention and before the feedforward sublayer, followed by a final normalization before the output head. :contentReference[oaicite:6]{index=6}

### Appendix D.1 RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(mean_sq + self.eps)
        return x * inv * self.weight
```

### Appendix D.2 MeanAbsNorm

```python
class MeanAbsNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.c = math.sqrt(math.pi / 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().mean(dim=-1, keepdim=True) * self.c
        return x / (scale + self.eps) * self.weight
```

### Appendix D.3 MeanAbsNorm-NoCorr

```python
class MeanAbsNormNoCorrection(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().mean(dim=-1, keepdim=True)
        return x / (scale + self.eps) * self.weight
```

### Appendix D.4 Normalization Factory
```python
def make_norm(kind: str, dim: int, eps: float, mm_alpha: float) -> nn.Module:
    kind = kind.lower()
    if kind == "layer":
        return ManualLayerNorm(dim, eps)
    if kind == "rms":
        return RMSNorm(dim, eps)
    if kind == "meanabs":
        return MeanAbsNorm(dim, eps)
    if kind == "meanabs_nocorr":
        return MeanAbsNormNoCorrection(dim, eps)
    if kind == "meanabs_fast1":
        return MeanAbsNormFastVariant1(dim, eps)
    if kind == "meanabs_fast2":
        return MeanAbsNormFastVariant2(dim, eps)
    if kind == "meanabs_fast3":
        return MeanAbsNormFastVariant3(dim, eps)
    if kind == "maxabs":
        return MaxAbsNorm(dim, eps)
    if kind == "maxmean":
        return MaxMeanNorm(dim, mm_alpha, eps)
    if kind == "none":
        return IdentityNorm(dim, eps)
    raise ValueError(f"unknown norm kind: {kind}")
```

### Appendix D.5 Transformer Block Placement
```python
class Block(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.n1 = make_norm(cfg.norm_kind, cfg.d_model, cfg.norm_eps, cfg.mm_alpha)
        self.attn = CausalSelfAttention(cfg)
        self.n2 = make_norm(cfg.norm_kind, cfg.d_model, cfg.norm_eps, cfg.mm_alpha)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ff(self.n2(x))
        return x
```

### Appendix D.6 Final Normalization
```python
class TinyLM(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = make_norm(cfg.norm_kind, cfg.d_model, cfg.norm_eps, cfg.mm_alpha)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        bsz, seqlen = idx.shape
        pos = torch.arange(seqlen, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
```

## Appendix E: Statistical Test Outputs

## Appendix F: Metadata and File-Path Audit

This appendix records metadata issues, file-path consistency checks, and repository hygiene notes that affect reproducibility.

### Appendix F.1 Current Corpus Coverage

The current repository contains completed result files for:

- *The City of God*
- *Anne of Green Gables*

The Sherlock Holmes corpus is reserved for the next completed run set.

### Appendix F.2 Known Metadata Issue

The Anne of Green Gables result files should be checked for internal path-label consistency. Some generated metadata or curve paths may retain `cityofgod` labels from the earlier output configuration even when the files are stored under the Anne corpus folder.

Before publication, verify that:

- `raw_csv` paths point to the correct corpus folder
- `summary_csv` paths point to the correct corpus folder
- `curve_path` fields point to the correct corpus folder
- plot paths match the corpus being reported
- corpus names in tables match the actual source file used

### Appendix F.3 Planned Corpus Expansion

The current report covers two completed corpora, with Sherlock Holmes pending. Additional corpora may be added in later versions of the study.
