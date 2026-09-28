# DASE7506 MP1 — Small Language Model Challenge

## 1. Final Result

The final submitted model is a small causal Transformer language model trained from random initialization on the supplied WikiText-2 training split.

The final model introduces two architectural changes to the supplied baseline:

1. **SwiGLU feed-forward networks** replace the baseline GELU MLP.
2. **Rotary Position Embeddings (RoPE)** replace learned absolute positional embeddings.

The final model was trained for 2,400 optimization steps.

### Final Results

| Metric | Result |
|---|---:|
| Validation BPB | 1.7417933 |
| Test BPB | **1.7666105** |
| Parameters | 1,069,312 |
| Training steps | 2,400 |
| Processed training targets | 19,660,800 |
| Seed | 17 |
| Context length | 256 |
| Test precision | FP32 |
| Test device | CPU |

Final checkpoint SHA256:

`26c326746c7d457174e5b20dadfbaf3e975fe626739f738dac1efaf70bb40721`

Final implementation SHA256:

`a13d3c0f29e588c7d70f0da02822f132f7976be5ce380c789cddb7837b56cba1`

---

## 2. Environment

The model was developed and evaluated with:

- Python 3.12
- PyTorch 2.7.1+cpu
- CPU evaluation
- 4 CPU threads
- FP32 ranked evaluation

Install the CPU version of PyTorch:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the supplied correctness tests:

```bash
python -m unittest discover -s tests -v
```

All supplied contract tests should pass before training or evaluation.

---

## 3. Model

The supplied baseline is a four-layer causal GPT with:

- vocabulary size: 2048
- width: 128
- attention heads: 4
- depth: 4
- context length: 256

My final model preserves the overall Transformer structure but makes two main architectural changes.

### 3.1 SwiGLU

The baseline GELU feed-forward network is replaced with a SwiGLU feed-forward network.

For model width \(d=128\), the SwiGLU hidden dimension is selected using approximately \(8d/3\) and rounded upward to a multiple of 16, giving a hidden dimension of 352.

This keeps the feed-forward parameter count close to that of the baseline while introducing multiplicative gating.

### 3.2 Rotary Position Embeddings

The learned absolute positional embedding is removed.

RoPE is applied to the query and key representations before causal scaled dot-product attention. Value representations are not rotated.

Removing the learned positional embedding also reduces the total parameter count. The final model contains 1,069,312 parameters, compared with 1,088,256 parameters in the supplied baseline.

The final model implementation is:

```text
student.py
```

---

## 4. Training

To reproduce the final model from random initialization, run the following command from the repository root:

```bash
python train.py --implementation student --device cpu --threads 4 --seed 17 --steps 2400 --run-dir runs/e03-rope-2400
```

With batch size 32 and context length 256, the final run processes:

```text
2400 × 32 × 256 = 19,660,800
```

training targets.

Training from scratch produces a checkpoint under:

```text
runs/e03-rope-2400/checkpoint.pt
```

For evaluation without retraining, the repository includes the frozen final checkpoint at:

```text
final_checkpoint/checkpoint.pt
```

The included checkpoint corresponds to the final submitted `student.py` implementation.

Training time depends on hardware and system load and is not required for evaluation reproduction.

---

## 5. Evaluation

### 5.1 Validation

Evaluate the included frozen final model on the validation split:

```bash
python evaluate.py --checkpoint final_checkpoint/checkpoint.pt --device cpu --precision fp32 --split validation
```

Expected validation BPB:

```text
1.7417933121880347
```

### 5.2 Complete Test

Evaluate the frozen final model on the complete test split:

```bash
python evaluate.py --checkpoint final_checkpoint/checkpoint.pt --device cpu --precision fp32 --split test
```

The complete test evaluation produced:

```text
BPB = 1.7666105432985555
token PPL = 40.16493098479789
targets = 428405
UTF-8 bytes = 1292013
```

The reported leaderboard value is the complete-test BPB.

The repository also includes the saved final-run metadata and test summary:

```text
final_checkpoint/metrics.json
final_checkpoint/test_cpu_fp32.json
```

---

## 6. Controlled Experiments

Validation was used for model development and model selection.

| Run | Method | Steps | Processed Targets | Parameters | Validation BPB |
|---|---|---:|---:|---:|---:|
| E00 | Baseline GPT | 1,200 | 9,830,400 | 1,088,256 | 2.0710835 |
| E01 | + SwiGLU | 1,200 | 9,830,400 | 1,102,080 | 2.0004562 |
| E02 | + SwiGLU + RoPE | 1,200 | 9,830,400 | 1,069,312 | 1.8369916 |
| E03 | + SwiGLU + RoPE, longer training | 2,400 | 19,660,800 | 1,069,312 | **1.7417933** |

E00, E01, and E02 use exactly the same number of processed training targets. They therefore provide the main controlled comparison of the architectural modifications under the same training-target budget.

Relative to E00, E01 reduces validation BPB by approximately 3.41%.

Relative to E01, E02 reduces validation BPB by approximately 8.17%.

Across the equal-budget E00-to-E02 comparison, validation BPB decreases from 2.0710835 to 1.8369916, corresponding to an improvement of approximately 11.30%.

E03 uses the same architecture and parameter count as E02 but doubles the number of processed training targets. It is therefore treated separately as a training-budget experiment rather than as an equal-budget architectural comparison.

Increasing training from E02 to E03 reduces validation BPB by approximately 5.18%.

The full experiment log is available in:

```text
RUN_LOG_TEMPLATE.csv
```

---

## 7. Resource Measurements

The final frozen model was evaluated on CPU using FP32.

### 7.1 CPU Scoring Time

On the same development machine:

| Model | Complete-Test CPU Scoring Time |
|---|---:|
| Baseline | 18.1351 s |
| Final model | 20.6256 s |

The final model therefore requires approximately:

```text
20.6256 / 18.1351 ≈ 1.14×
```

the baseline CPU scoring time on the same machine.

This is below the coursework limit of 5× baseline CPU scoring time.

### 7.2 Peak RAM

Peak CPU RAM during complete-test evaluation was measured at approximately:

```text
1.560 GiB
```

This is below the 4 GiB evaluation RAM limit.

The auxiliary resource-measurement script used for this check is:

```text
measure_ram.py
```

It does not modify the supplied evaluator or model predictions.

### 7.3 Inference Asset Size

The frozen final checkpoint size is:

```text
4,293,621 bytes
```

or approximately:

```text
4.10 MiB
```

This is below the 64 MiB inference-asset limit.

---

## 8. Test-Set Use

All model development, architectural decisions, and model selection used the validation split.

The final SwiGLU + RoPE architecture and the 2,400-step training configuration were selected using validation performance.

The final method was frozen before complete-test evaluation.

The complete test split was then used to obtain the final reported BPB and for repeated reproduction and resource measurements of the same frozen predictor only.

No architecture, hyperparameter, or checkpoint selection was performed using the final test result.

---

## 9. Repository Structure

The main submission files are organized as follows:

```text
.
├── configs/
├── data/
├── final_checkpoint/
│   ├── checkpoint.pt
│   ├── metrics.json
│   └── test_cpu_fp32.json
├── report/
│   ├── DASE7506_MP1_Report.docx
│   └── DASE7506_MP1_Report.pdf
├── tests/
├── common.py
├── evaluate.py
├── measure_ram.py
├── model.py
├── requirements.txt
├── RUN_LOG_TEMPLATE.csv
├── student.py
├── student_e01_swiglu.py
├── student_original.py
├── train.py
├── README.md
└── SUBMISSION_README.md
```

`student.py` contains the final submitted architecture.

`model.py` contains the supplied baseline implementation.

`student_e01_swiglu.py` is retained as the SwiGLU-only ablation implementation.

`final_checkpoint/checkpoint.pt` is the frozen final checkpoint used for the reported complete-test result.

The `runs/` directory contains local development runs and is excluded from Git.

---

## 10. Report

The coursework report is provided in:

```text
report/DASE7506_MP1_Report.pdf
```

An editable Word version is also included:

```text
report/DASE7506_MP1_Report.docx
```

The report describes the baseline, proposed architecture, controlled experiments, training-budget experiment, final test result, resource measurements, limitations, and critical analysis.

---

## 11. AI Assistance Disclosure

Generative AI assistance was used during development for:

- explaining the supplied baseline implementation and coursework requirements;
- discussing possible architectural modifications;
- assisting with the implementation and explanation of SwiGLU and RoPE;
- interpreting experimental results;
- checking experiment organization and reproducibility requirements;
- assisting with documentation and report drafting.

The model implementation, experiments, outputs, and final submission were reviewed and executed by the student.

The student is responsible for understanding and explaining the submitted method and results.