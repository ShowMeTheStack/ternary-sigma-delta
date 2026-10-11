# ternary-sigma-delta

Code for training neural networks with binary and ternary weights through the straight-through estimator (STE) and for studying how the quantized weights move during training. During training each quantized weight switches between neighbouring levels. The code records, per weight, how often each level is visited over a window of steps (the duty cycle) and how often the level changes (the flip rate). From these counts it builds alternative deployed weights: the last snapshot, the most-visited level (mode, or duty-cycle rounding), the time average of the levels (dither-free), and a requantized exponential moving average of the latent weights. The repository contains a language-model trainer (a small Llama-style decoder trained on FineWeb-Edu or OpenWebText), a cool-down probe and other post-training analysis tools, a ResNet-18 trainer for CIFAR-10 and CIFAR-100, and synthetic per-coordinate experiments on quadratic losses.

## Contents

- [Requirements](#requirements)
- [Repository layout](#repository-layout)
- [Data preparation](#data-preparation)
- [Training language models](#training-language-models)
- [Post-training tools](#post-training-tools)
- [CIFAR experiments](#cifar-experiments)
- [Synthetic experiments](#synthetic-experiments)
- [Hardware, run time and interruptions](#hardware-run-time-and-interruptions)
- [Licence](#licence)

## Requirements

- Python 3.10 or newer.
- PyTorch with CUDA for the language-model and CIFAR scripts. The scripts use bf16 autocast, `torch.compile` (can be switched off in `train.py`) and fused AdamW.
- NumPy.
- tiktoken (GPT-2 tokenizer) for data preparation and zero-shot evaluation.
- Hugging Face `datasets` for `data/prepare_owt.py` and `analysis/zeroshot.py`. `data/prepare.py` uses `huggingface_hub` and `pyarrow`, which are installed together with `datasets`.
- matplotlib, only for the optional `--plot` output of `analysis/rice.py`.

The code was run with PyTorch 2.11.0, CUDA 12.8, NumPy 2.5.3 and tiktoken 0.14.0.

```bash
pip install torch numpy tiktoken datasets matplotlib
```

All commands below are run from the repository root. The scripts add the repository root to `sys.path` themselves, so no installation step is needed.

## Repository layout

| File | What it does |
|---|---|
| `train.py` | Language-model trainer with STE-quantized linear layers, duty-cycle counting, cool-down and final deployment evaluations. |
| `quant/ste.py` | `QuantLinear` and `QuantConv2d`: binary, ternary, uniform and custom-level quantizers, scale rules (absmean, LSQ, TWN), oscillation dampening, iterative freezing, level counts, flip counts and weight overrides. |
| `model/llama_small.py` | Small decoder (RMSNorm, SwiGLU, RoPE, with an optional GPT-2 style variant) built from `QuantLinear`, and the model sizes `SIZES`. |
| `data/prepare.py` | Downloads FineWeb-Edu sample-10BT parquet files and writes GPT-2 token shards. |
| `data/prepare_owt.py` | Streams OpenWebText and writes GPT-2 token shards. |
| `cifar/train_cifar.py` | ResNet-18 on CIFAR-10 or CIFAR-100 with STE-quantized convolutions, cool-down and deployment evaluations with and without batch-norm recalibration. |
| `analysis/probe.py` | Cool-down probe: continues a finished run for K steps at the final learning rate with counting on, then evaluates the deployment rules. Resumable. |
| `analysis/zeroshot.py` | Zero-shot LAMBADA, HellaSwag, PIQA, ARC-Easy and WinoGrande for snapshot, mode and EMA weights. |
| `analysis/perbatch_eval.py` | Per-batch validation losses of a finished run, standard errors and autocorrelation. Resumable. |
| `analysis/predict_gamma.py` | Per-weight penalty curves G_T, G_B, M_T over the scale multiplier gamma from the weights of a BF16 run, and their minimizers. |
| `analysis/rice.py` | Level-crossing (Rice) prediction of the per-layer flip rate against the measured flip rate. Optional scatter plot. |
| `analysis/residual_share.py` | Signal power entering each normalization layer and the share contributed by the attention and MLP branches. |
| `analysis/attribute.py` | Re-runs a short cool-down and applies dither-free weights to subsets of layers (all, attention, MLP, single blocks). |
| `analysis/rescale_test.py` | Evaluates a run with the quantized weights multiplied by constant factors. |
| `analysis/scale_test.py` | Table of measured dither gap and mode gain across a gamma sweep next to the curves from `predict_gamma.py`. |
| `analysis/fixed_target.py` | Simulated duty-cycle snapshots and nearest-level rounding of a fixed weight set, evaluated over gamma. |
| `analysis/sr_predict.py` | Predicted and measured dither variance of stochastic rounding against deterministic rounding. |
| `analysis/mode_ratio.py` | Predicted ratio of mode gain to dither gap from the duty cycles against the measured ratio. |
| `analysis/lm_summary.py` | Summary table of all runs under a directory. Optional dither variance and flip-rate curves. |
| `analysis/curves.py` | Short text summary of training and validation curves from `log.jsonl`. |
| `analysis/cifar_reeval.py` | Re-evaluates a finished CIFAR run with batch-norm statistics re-estimated on augmented training data. |
| `synth/e1.py` | Per-coordinate quadratic-loss experiment with SGD and Adam, binary and ternary quantizers, compared with closed forms. |
| `synth/e1_noise.py` | The same setting with Gaussian gradient noise. |
| `synth/e1_noise_gamma.py` | Dither gap against the scale s under gradient noise. |
| `synth/adam_relay_1d.py` | One-dimensional deterministic Adam relay: flip rate and period of the locked cycle. |
| `synth/adamw_slope_drop.py` | Predicted change of the duty-cycle slope under decoupled AdamW weight decay. |
| `synth/dither_lloyd.py` | Dither-optimal k-level quantizers compared with Lloyd-Max levels. |
| `synth/dlloyd_check.py` | STE-Adam simulation with the dither-optimal and Lloyd-Max levels. |
| `synth/momentum_window.py` | Finite-window error of the duty-cycle average for SGD and heavy-ball momentum. |
| `synth/window_rice_check.py` | Finite-window error under Adam and a level-crossing check of the flip rate in simulation. |

## Data preparation

Both preparation scripts tokenize with the GPT-2 tokenizer (tiktoken), append the end-of-text token (50256) after every document and write flat `uint16` memmap files of 100M tokens each. The first file written is the validation shard `val.bin`, the following ones are `train_0000.bin`, `train_0001.bin`, ... The trainer reads `val.bin` for validation and every `train_*.bin` for training, so a directory produced by either script can be passed directly as `--data`.

### FineWeb-Edu

```bash
python data/prepare.py --out ~/data/fineweb_edu_gpt2 --files 6 --workers 30
```

| Option | Default | Meaning |
|---|---|---|
| `--out` | required | Output directory for the shards. |
| `--files` | `6` | Process parquet files `sample/10BT/000_00000.parquet` up to (not including) this index from the `HuggingFaceFW/fineweb-edu` dataset repository. |
| `--workers` | `8` | Tokenizer processes. |
| `--first_file` | `0` | First parquet file index to process (for resuming). |
| `--start_shard` | `0` | Shard index to start writing at. 0 is `val.bin`, 1 is `train_0000.bin` (for resuming). |
| `--cache` | `~/data/hf_cache` | Hugging Face download cache for the parquet files. |

Six parquet files give about 4.3B tokens, which is about 8.6 GB of shards (2 bytes per token), plus the downloaded parquet files in the cache directory. To resume after an interruption, restart with `--first_file` set to the file that was being processed and `--start_shard` set to the index of the first shard that was not finished. The partial token buffer of the interrupted file is not saved, so that file is tokenized again.

### OpenWebText

```bash
python data/prepare_owt.py              # 1.2B tokens (default target)
python data/prepare_owt.py 2000000000   # custom token target
```

The script streams `Skylion007/openwebtext` with `datasets` (streaming mode, `trust_remote_code=True`), uses 8 tokenizer processes, and stops once the token target (single optional positional argument, default 1,200,000,000) is reached. The check happens after every block of 2048 documents. The output directory is fixed in the script to `~/data/openwebtext_gpt2`. The default target gives about 2.4 GB of shards.

### CIFAR

`cifar/train_cifar.py` downloads the python version of CIFAR-10 or CIFAR-100 from the University of Toronto site into its `--data` directory on first use. No torchvision is needed.

## Training language models

`train.py` trains a decoder from scratch with AdamW (betas 0.9 and 0.95, eps 1e-8, weight decay on matrices only), gradient clipping at 1.0, bf16 autocast and a linear warm-up followed by a cosine decay to `lr * min_lr_frac`. Every `QuantLinear` (the attention and MLP projections) is quantized in the forward pass and receives the straight-through gradient. The embedding, the output head (untied) and the norms stay in high precision.

Training batches are random windows drawn with a NumPy generator seeded by `--seed`, so two runs with the same seed see the same data in the same order whatever their quantizer settings.

### Model sizes

Sizes come from `SIZES` in `model/llama_small.py`. All use a head dimension of 64, a vocabulary of 50304 and a SwiGLU hidden size of `round(8d/3/64)*64`. The parameter counts below are for the default `--arch llama` and exclude the embedding and the output head (which together add `2 * 50304 * d`).

| `--size` | d | layers | heads | MLP hidden | quantized parameters |
|---|---|---|---|---|---|
| `XS` | 128 | 2 | 2 | 320 | 0.4M (smoke tests) |
| `S` | 384 | 6 | 6 | 1024 | 10.6M |
| `M` | 512 | 8 | 8 | 1344 | 24.9M |
| `L` | 768 | 12 | 12 | 2048 | 84.9M |
| `XL16` | 1024 | 16 | 16 | 2752 | 202.4M |
| `XL` | 1024 | 24 | 16 | 2752 | 303.6M |

With `--arch gpt2` the blocks use LayerNorm, a GELU MLP of width 4d (two matrices instead of three), learned position embeddings and no RoPE.

### Quantizer options

| Option | Default | Meaning |
|---|---|---|
| `--alphabet` | `ternary` | `ternary`: levels {-s, 0, +s}. `binary`: {-s, +s} (sign, with 0 mapped to +s). `none`: no quantization (BF16 autocast baseline). `uniform_k`: k symmetric levels with k odd, step `gamma * max|W| / ((k-1)/2)`, plain rounding. `levels:a,b,...`: custom symmetric levels `+-a*s, +-b*s, ...` without a zero level and with nearest-level rounding, for example `levels:0.5,1.5` gives four levels. |
| `--gamma` | `1.0` | Scale multiplier: s = gamma * mean\|W\| per tensor (absmean rule), and the initial value of s under LSQ. |
| `--dz` | `0.5` | Ternary dead zone: a weight is zero when \|W\| <= dz * s (absmean and LSQ rules). gamma = 1, dz = 0.5 is the BitNet b1.58 quantizer. |
| `--scale_rule` | `absmean` | `absmean`: s = gamma * mean\|W\|, detached. `lsq`: one learned scale per tensor (LSQ gradient, initialised at gamma * mean\|W\|). `twn`: threshold 0.7 * mean\|W\| and level equal to the mean of \|W\| above the threshold. Ternary only. |
| `--stochastic` | off | Stochastic instead of deterministic rounding (ternary alphabet). |
| `--act_bits` | none | `8` quantizes the inputs of every quantized layer to int8 with a per-token absmax scale (STE). Default keeps bf16 activations. |

Level counting and the duty-cycle evaluations are done for binary, ternary, `uniform_3` and custom-level layers.

### Optimization and schedule

| Option | Default | Meaning |
|---|---|---|
| `--tokens` | `0.6e9` | Training tokens of the schedule. The number of optimizer steps is `tokens // batch_tokens`. |
| `--batch_tokens` | `524288` | Tokens per optimizer step. |
| `--micro_bs` | `32` | Sequences per micro-batch. Gradient accumulation is `batch_tokens // (micro_bs * seq)`, so choose `micro_bs` such that `micro_bs * seq` divides `batch_tokens`. Also used as the evaluation batch size. |
| `--seq` | `1024` | Sequence length. |
| `--lr` | `1.2e-3` | Peak learning rate. |
| `--min_lr_frac` | `0.1` | Final learning rate as a fraction of the peak. |
| `--warmup_frac` | `0.02` | Warm-up length as a fraction of the schedule ... |
| `--min_warmup` | `100` | ... but at least this many steps. |
| `--wd` | `0.1` | AdamW weight decay on parameters with two or more dimensions. |
| `--wd_off_frac` | `0.0` | Switch weight decay off for the last fraction of all steps (including the cool-down). |
| `--seed` | `0` | Seed for parameters and for the data order. |
| `--max_steps` | none | Override the schedule length in steps (for debugging). |
| `--cooldown` | `0` | Extra steps after the schedule at the constant final learning rate with the live optimizer. If positive, the counting window is the cool-down. Otherwise it is the last `--final_window` steps of the schedule. |
| `--final_window` | `64` | Counting window when `--cooldown 0`. |

The counting window starts `window` steps before the end of training. From then on every counted layer records per-weight level visit counts and flip counts (int16, so keep the window below 32768 steps), the level trajectory of 1024 fixed weights per layer, the mean absolute latent step and an EMA of the latent weights with beta = 1 - 2 / window.

### Oscillation dampening and iterative freezing

Two baselines for weight oscillations, both off by default and applied to the counted layers:

| Option | Default | Meaning |
|---|---|---|
| `--osc_damp` | `0.0` | Dampening weight lambda: adds the gradient of `lambda * sum (w - q)^2` to the latent gradient, where q is the current grid point. |
| `--damp_start_frac` | `0.5` | lambda ramps linearly from 0 at this fraction of the schedule to its full value at the end of the schedule. |
| `--freeze_fth` | `0.0` | Iterative freezing threshold on the per-weight EMA of the flip frequency, and 0 turns it off. A weight above the threshold is frozen at the level nearest the EMA of its level index. Its gradient is zeroed and its latent is pinned. |
| `--freeze_start_frac` | `0.2` | Fraction of the schedule at which freezing starts. |
| `--freeze_momentum` | `0.99` | EMA momentum for the flip frequency and the level index. |

### Logging, evaluation, checkpoints and other options

| Option | Default | Meaning |
|---|---|---|
| `--data` | `~/data/fineweb_edu_gpt2` | Shard directory (`val.bin`, `train_*.bin`). |
| `--out` | required | Run directory. |
| `--arch` | `llama` | `llama` or `gpt2`. |
| `--log_every` | `10` | Steps between records in `log.jsonl`. |
| `--hist_every` | `100` | Steps between latent histograms in `hist.jsonl`. |
| `--eval_every` | `250` | Steps between validation evaluations. |
| `--eval_batches` | `40` | Validation micro-batches per periodic evaluation. |
| `--final_eval_batches` | `400` | Validation micro-batches for the final evaluations. |
| `--ckpt_every` | `100` | Steps between checkpoints. |
| `--compile` | `1` | `torch.compile` the model for training (0 to switch off). |
| `--device` | `cuda` | Torch device, for example `cuda` or `cuda:1`. |
| `--lean` | off | Memory-lean cool-down for large models: no latent EMA (so no EMA evaluation), no latent-step telemetry, uncompiled evaluation and a loss chunk of one sequence. |

Validation always uses the first consecutive windows of `val.bin` (`final_eval_batches * micro_bs * seq` tokens for the final evaluations), so all runs with the same `micro_bs` and `seq` are evaluated on the same tokens.

### Outputs written into `--out`

| File | Contents |
|---|---|
| `args.json` | All command-line arguments. The analysis scripts read the model and data settings from it. |
| `ckpt.pt` | Checkpoint written every `--ckpt_every` steps before the counting window: model and optimizer state, next step, data-generator state and freezing state. Written atomically (temporary file, then rename). Deleted when the run finishes. |
| `log.jsonl` | One JSON record per logged step: `step`, `tokens`, `lr`, `loss` (training loss of the step), `gnorm` (gradient norm before clipping), `tps` (tokens per second, null after a resume), `elapsed`, and for quantized runs `flip` / `zero` / `scale` (per layer: mean flip rate since the last record, zero fraction, scale), `flip_mean`, `zero_mean`, plus `scale_ratio` (LSQ), `frozen_mean` (freezing) and `lam` (dampening) when active. Separate records `{"step", "val_loss"}` for periodic validation and, with a cool-down, `{"step", "val_snapshot_end_of_schedule"}`. |
| `hist.jsonl` | Every `--hist_every` steps: `{"step", "hist": {layer: 64-bin histogram of w / mean\|w\| on [-4, 4], normalized to sum 1}}`. |
| `final.json` | `n_params` (non-embedding), `steps`, `sched_steps`, `cooldown`, `tokens`, `elapsed`, `val_snapshot` and, with a cool-down, `val_snapshot_end_of_schedule`. For quantized runs also `val_mode` (most-visited level), `val_dither_free` (time-averaged levels), `val_ema_requant` (requantized latent EMA, null in lean mode), `val_latent_fp` (unquantized latent weights), `mean_zero_frac`, `mean_flip_rate_window`, `mean_frozen_frac`, and `per_layer` with `scale`, `zero_frac`, `zero_frac_from_duty`, `flip_rate_window`, `frac_two_level`, `frac_three_level`, `frac_static`, `mode_eq_snapshot`, `ema_eq_mode`, `dither_var_proxy`, `mean_abs_step`, `scale_ratio`, `frozen_frac`. |
| `final_state.pt` | For quantized runs a dict with one entry per counted layer (`latent` fp16, `counts` int16 of shape `(levels, numel)`, `flipc` int16, `traj` int8 trajectory of 1024 fixed weights, `traj_idx`, `ema` fp16 or null) and `_other` with every other parameter in fp16. For `--alphabet none` a flat dict of all parameters in fp16. |

### Examples

Small ternary run (S, 0.6B tokens, 256-step cool-down at the final learning rate):

```bash
python train.py --size S --alphabet ternary --gamma 1.0 --dz 0.5 --tokens 0.6e9 \
    --lr 1.2e-3 --seed 0 --cooldown 256 --data ~/data/fineweb_edu_gpt2 --out runs/S_ter_g1.0_s0
```

Binary run with the same seed and therefore the same data order:

```bash
python train.py --size S --alphabet binary --seed 0 --cooldown 256 --out runs/S_bin_g1.0_s0
```

BF16 baseline (no quantization):

```bash
python train.py --size S --alphabet none --seed 0 --out runs/S_fp_s0
```

Four-level custom alphabet (levels -1.5s, -0.5s, +0.5s, +1.5s):

```bash
python train.py --size S --alphabet levels:0.5,1.5 --seed 0 --cooldown 256 --out runs/S_lv4_s0
```

Learned (LSQ) and TWN scale rules:

```bash
python train.py --size S --alphabet ternary --scale_rule lsq --seed 0 --cooldown 256 --out runs/S_ter_lsq_s0
python train.py --size S --alphabet ternary --scale_rule twn --seed 0 --cooldown 256 --out runs/S_ter_twn_s0
```

Scale multiplier sweep point, stochastic rounding, int8 activations and the GPT-2 style block:

```bash
python train.py --size S --alphabet ternary --gamma 1.25 --seed 0 --out runs/S_ter_g1.25_s0
python train.py --size S --alphabet ternary --stochastic --seed 0 --out runs/S_ter_g1.0_stoch_s0
python train.py --size S --alphabet ternary --act_bits 8 --seed 0 --out runs/S_ter_a8_s0
python train.py --size S --alphabet ternary --arch gpt2 --seed 0 --out runs/S_ter_gpt2_s0
```

Oscillation dampening and iterative freezing:

```bash
python train.py --size S --alphabet ternary --osc_damp 1e-4 --seed 0 --cooldown 256 --out runs/S_ter_damp_s0
python train.py --size S --alphabet ternary --freeze_fth 0.02 --seed 0 --cooldown 256 --out runs/S_ter_frz_s0
```

Large model in lean mode with a small micro-batch:

```bash
python train.py --size XL16 --alphabet ternary --micro_bs 2 --lean --cooldown 256 --out runs/XL16_ter_s0
```

The numeric values in these examples (cool-down length, dampening weight, freezing threshold) are illustrations, not recommendations.

## Post-training tools

All tools take a run directory written by `train.py` (or by `cifar/train_cifar.py` for `cifar_reeval.py`) and read `args.json` and `final_state.pt` from it. Unless stated otherwise, results go into the run directory.

Restrictions to keep in mind:

- `probe.py`, `attribute.py`, `rescale_test.py` and `fixed_target.py` rebuild the model with the default absmean scale rule. Use them on absmean runs.
- The mode weights in `probe.py`, `perbatch_eval.py` and `zeroshot.py` and the per-layer statistics of `probe.py`, `lm_summary.py --curves` and `mode_ratio.py` assume three level rows (binary or ternary).
- `probe.py`, `residual_share.py` and `cifar_reeval.py` pass `--device` directly to `torch.autocast`. Use `cuda` or `cpu` and select the GPU with `CUDA_VISIBLE_DEVICES`. `attribute.py`, `rescale_test.py`, `fixed_target.py` and `zeroshot.py` always use `cuda`.

### Cool-down probe: `analysis/probe.py`

Reloads the final weights of a quantized run, trains `--steps` more steps at a constant learning rate (default: the run's final learning rate `lr * min_lr_frac`, with weight decay as in the run, or 0 if the run used `--wd_off_frac`) with level counting on for every step, and then evaluates the pre-probe snapshot, the post-probe snapshot, mode-rounded, dither-free and EMA-requantized weights (EMA beta = 1 - 2 / K).

```bash
python analysis/probe.py runs/S_ter_g1.0_s0 --steps 64
```

| Option | Default | Meaning |
|---|---|---|
| `run` | | Run directory (positional). |
| `--steps` | `64` | Probe length K. |
| `--lr` | run `lr * min_lr_frac` | Probe learning rate. |
| `--eval_batches` | `400` | Validation micro-batches per evaluation. |
| `--data` | from `args.json` | Shard directory. |
| `--save_bars` | off | Also save the dither-free weights to `bars_K.pt`. |
| `--ckpt_every` | `100` | Save `probe_K_ckpt.pt` every N steps (0 = off). On restart the probe resumes from it, with counts, flip counts and EMA restored. It is deleted at the end. |
| `--device` | `cuda` | Device. |

Output `probe_K.json`: `val_pre_snapshot`, `val_post_snapshot`, `val_mode`, `val_dither_free`, `val_ema_requant`, `probe_seconds`, learning rate and weight decay used, `per_layer` statistics (`zero_frac`, `frac_static`, `frac_two`, `frac_three`, `flip_rate`, `var_q_over_s2`, `mode_eq_snapshot`, `ema_eq_mode`) and their means (`mean_*`).

### Zero-shot evaluation: `analysis/zeroshot.py`

```bash
python analysis/zeroshot.py runs/L_ter_g1.0_s0 --which snapshot mode ema --n 2000 \
    --tasks lambada hellaswag piqa arc_easy winogrande
```

| Option | Default | Meaning |
|---|---|---|
| `run` | | Run directory (positional). |
| `--which` | `snapshot mode` | Weight variants: `snapshot` (final quantized weights), `mode` (most-visited level in the counting window), `ema` (requantized latent EMA, skipped if the run has none). A BF16 run is evaluated once as `snapshot`. |
| `--n` | `2000` | Maximum number of examples per task. |
| `--tasks` | `lambada hellaswag` | Any of `lambada`, `hellaswag`, `piqa`, `arc_easy`, `winogrande`. |
| `--out` | `zeroshot.json` | Output file name in the run directory. |

| Task | Source (Hugging Face) | Split | Scoring |
|---|---|---|---|
| `lambada` | `EleutherAI/lambada_openai` (`en`) | test | Greedy decoding of the last word. Exact token match. |
| `hellaswag` | `Rowan/hellaswag` | validation | 4-way, log-likelihood of each ending divided by its token count. |
| `piqa` | `ybisk/piqa` (parquet revision) | validation | 2-way, length-normalized log-likelihood. |
| `arc_easy` | `allenai/ai2_arc` (`ARC-Easy`) | test | Prompt `Question: ... Answer:`, length-normalized log-likelihood. |
| `winogrande` | `allenai/winogrande` (`default`, parquet revision) | validation | The option fills the blank. The log-likelihood of the text after the blank is compared (partial scoring). Duplicate sentences are skipped. |

The output holds accuracies as `{variant: {task: accuracy}}` and is rewritten after every task. Existing entries in the file are kept, and listed tasks are recomputed.

### Per-batch validation losses: `analysis/perbatch_eval.py`

Evaluates the same validation micro-batches as the final evaluation of `train.py` (first consecutive windows of `val.bin`, `micro_bs` and `seq` from `args.json`) and keeps the loss of every micro-batch.

```bash
python analysis/perbatch_eval.py runs/S_ter_g1.0_s0 --which snapshot mode --device cuda:0
```

| Option | Default | Meaning |
|---|---|---|
| `run` | | Run directory (positional). |
| `--device` | `cuda` | Device. |
| `--which` | `snapshot` | `snapshot` and/or `mode`. |
| `--n_batches` | `final_eval_batches` from `args.json` (400) | Number of micro-batches. |
| `--data` | from `args.json` | Shard directory. |
| `--out` | `perbatch_eval.json` | Output file name in the run directory. |
| `--loss_chunk` | 1 for lean runs, else model default (8) | Sequences per head and cross-entropy chunk. 1 lowers memory for large models. |

Output: `meta` (run, size, batch shape, token count, validation file), and per variant `per_batch` losses with `mean`, `std_batch`, `se` (standard error of the batch means), `lag1_autocorr`, `se_blocked` (from means of 20 consecutive batches), plus `mode_minus_snapshot` statistics of the paired difference when both variants are evaluated and `final_json` with the corresponding values from `final.json`. The per-batch list is saved every 25 batches and reused on restart.

### Penalty curves over gamma: `analysis/predict_gamma.py`

Reads the final weights of a BF16 run (`--alphabet none`) and, for every block weight matrix with |W| = u and s = gamma * E|W|, computes

- G_T(gamma) = mean of u(s - u) for u < s and (u - s)^2 otherwise (ternary snapshot),
- G_B(gamma) = mean of s^2 - u^2 for u < s and (u - s)^2 otherwise (binary snapshot),
- M_T(gamma) = mean of min(u^2, (s - u)^2) for u < s and (u - s)^2 otherwise (ternary mode rounding),

normalized by (E|W|)^2 per layer and pooled over layers weighted by size.

```bash
python analysis/predict_gamma.py runs/S_fp_s0 [--gammas 0.5 0.75 1.0 ...]
```

`--gammas` defaults to 0.50 to 2.50 in steps of 0.05. Output `predict_gamma.json`: `n_weights`, `mean_excess_kurtosis`, the minimizing gammas `gamma_star_snapshot_ternary`, `gamma_star_snapshot_binary`, `gamma_star_mode_ternary`, `ratio_B_over_T_at_gamma1`, predicted zero fractions at gamma = 1 and at the ternary minimizer, and the full `curve`. Runs on CPU.

### Level-crossing flip-rate prediction: `analysis/rice.py`

For each layer of each run, predicts the flip rate in the noise-dominated regime as `2 * rho(theta) * delta_bar`, where rho is the latent density at the threshold (from the last histogram in `hist.jsonl`, in units of E|W|, summed over the thresholds +-dz*gamma for ternary, at 0 for binary) and `delta_bar = lr * min_lr_frac * sqrt((1 - 0.9) / (1 + 0.9)) / E|W|` is the typical Adam step for white-noise gradients. The measured value is `per_layer.flip_rate_window` from `final.json`, so the runs need `--cooldown > 0`.

```bash
python analysis/rice.py runs/M_ter_g1.0_s0 runs/M_bin_g1.0_s0 [--plot rice.pdf]
```

Prints, per run, the number of layers, mean predicted and measured rates, their ratio, a through-origin fit coefficient and the correlation, and the first four layers. With `--plot FILE` it also saves a scatter plot of predicted against measured per-layer flip rates for all runs (ternary filled markers, binary open markers, marker shape chosen from the run directory name). Requires matplotlib 3.7 or newer for the legend placement.

### Stream power per block: `analysis/residual_share.py`

Runs validation batches through the model with hooks on every block and measures the mean power of the residual stream entering the block (`S_in`), of the attention output (`y_attn`), of the stream entering the second norm (`S_attn`), of the MLP output (`y_mlp`) and of the block output (`S_mlp`), with the shares `mu_attn = y_attn / S_attn` and `mu_mlp = y_mlp / S_mlp`.

```bash
python analysis/residual_share.py runs/S_ter_g1.0_s0 runs/S_bin_g1.0_s0 --device cpu
```

Options: one or more run directories (positional), `--device` (`cuda`), `--batches` (`4`), `--bs` (`4` sequences per batch), `--data`. Writes `residual_share.json` into each run directory. With exactly two runs it also prints the mean ratio of the stream powers of the first to the second run over all blocks.

### Per-block attribution: `analysis/attribute.py`

Reloads a run's final weights, runs a fresh K-step cool-down at the final learning rate (seed + 2000) with level counting on, then evaluates the validation loss with dither-free weights applied to no layers (snapshot), all layers, attention only, MLP only, and each block on its own.

```bash
python analysis/attribute.py runs/S_ter_g1.0_s0 --steps 64 [--eval_batches 400]
```

Output `attribute_K.json` with those losses (`snapshot`, `all`, `attn`, `mlp`, `block0`, ...) and `two_level_frac` per layer.

### Rescaling test: `analysis/rescale_test.py`

Evaluates the snapshot weights multiplied by each factor in `--scales` (default `0.5 0.75 0.9 1.0 1.1 1.25 1.5 2.0`), and with a factor 2 applied only to the attention projections, only to the MLP, or only to the query and key projections.

```bash
python analysis/rescale_test.py runs/S_ter_g1.0_s0 [--eval_batches 400]
```

Output `rescale.json` with one validation loss per factor and `c2_attn`, `c2_mlp`, `c2_qk`.

### Gamma sweep table: `analysis/scale_test.py`

Collects all runs under `--root` with the given alphabet, dz = 0.5, deterministic rounding, no `--wd_off_frac` and bf16 activations, and prints per run the validation loss, zero fraction, the dither gap (probe snapshot minus dither-free) and mode gain (probe snapshot minus mode) from `probe_K.json`, the mean probe dither variance, and the curves G_T or G_B and M_T from `predict_gamma.json` of the BF16 run `--fp`, relative to gamma = 1. The predicted gap column scales the measured mean gap of the gamma = 1 runs by the relative curve.

```bash
python analysis/predict_gamma.py runs/S_fp_s0
for d in runs/gamma_sweep/*/; do python analysis/probe.py $d --steps 64; done
python analysis/scale_test.py --root runs/gamma_sweep --fp runs/S_fp_s0 --alphabet ternary --K 64
```

The run gammas must lie on the `--gammas` grid used by `predict_gamma.py`. Prints only.

### Fixed-target simulation: `analysis/fixed_target.py`

Takes a fixed weight set and, for each gamma, evaluates (a) a simulated duty-cycle snapshot, in which every weight is drawn from its two bracketing ternary levels with the probability that reproduces its value on average (mean and standard deviation over `--draws` draws), and (b) nearest-level rounding. The fixed weights are the BF16 run's weights, or with `--target` the dither-free weights saved by `probe.py --save_bars` for a ternary run (the BF16 run directory is still needed for the model and data settings).

```bash
python analysis/fixed_target.py runs/S_fp_s0 --gammas 0.5 0.75 1.0 1.25 1.5 2.0 2.5 --draws 3
python analysis/fixed_target.py runs/S_fp_s0 --target runs/S_ter_g1.0_s0/bars_64.pt --target_run runs/S_ter_g1.0_s0
```

Options: `fp_run` (positional), `--gammas` (default `0.5 0.75 1.0 1.25 1.5 2.0 2.5`), `--draws` (`3`), `--eval_batches` (`200`), `--target`, `--target_run`. Output `fixed_target_bf16.json` in the BF16 run, or `fixed_target_bars.json` in `--target_run`, with the unquantized loss (`target`) and per gamma `dither_mean`, `dither_std`, `rounded`, `zero_frac`.

### Stochastic rounding variance: `analysis/sr_predict.py`

Predicts the dither variance under stochastic rounding as E[u(1 - u)] with u = |w|/s over interior weights, from the last latent histogram of the stochastic run (gamma = 1 assumed), and compares the predicted and measured variance ratios with the measured ratio of loss gaps (probe snapshot minus dither-free) between the stochastic and the deterministic run. Both runs need `probe_64.json`.

```bash
python analysis/probe.py runs/S_ter_g1.0_stoch_s0 --steps 64
python analysis/probe.py runs/S_ter_g1.0_s0 --steps 64
python analysis/sr_predict.py runs/S_ter_g1.0_stoch_s0 runs/S_ter_g1.0_s0
```

Positional arguments: the stochastic run, then the deterministic run. Prints only.

### Mode gain to dither gap ratio: `analysis/mode_ratio.py`

From the counts in `final_state.pt`, over the weights that visited exactly two levels in the window, with p the share of the upper level, computes R = E[min(p, 1 - p)^2] / E[p(1 - p)] and prints it next to the measured ratio (snapshot - mode) / (snapshot - dither-free) from `final.json`. Runs without a cool-down are skipped.

```bash
python analysis/mode_ratio.py runs/M_ter_g1.0_s0 runs/M_bin_g1.0_s0
```

### Run summary: `analysis/lm_summary.py`

```bash
python analysis/lm_summary.py runs/sweep            # one line per run directory under runs/sweep
python analysis/lm_summary.py runs/sweep --curves   # also dither variance and flip-rate curves
```

Reads every subdirectory of the root that contains `final.json` (and `args.json`, `log.jsonl`) and prints alphabet, learning rate, gamma, the validation losses of the snapshot, mode, dither-free, EMA-requantized and latent weights, mean zero fraction, mean flip rate in the window, and the fractions of two-level and static weights. With `--curves` it also prints, per quantized run, the mean per-weight dither variance Var(q)/s^2 next to the measured loss gap, and the flip rate binned by the duty-cycle estimate |qbar|/s next to the noise-free values (1 - x for binary, 2 min(x, 1 - x) for ternary) and the fraction of steps at zero. Prints only.

### Training curves: `analysis/curves.py`

```bash
python analysis/curves.py runs/S_ter_g1.0_s0 runs/S_bin_g1.0_s0
```

For each run directory prints the training loss at steps 10, 30, 50, 100, 150, 200, 250 and 285, all validation losses, the mean flip rate at steps 100 and 250 and the mean zero fraction at step 250. Intended for short runs.

## CIFAR experiments

### Training: `cifar/train_cifar.py`

ResNet-18 with the CIFAR stem (3x3 convolution, stride 1, no max-pool). All convolutions in the residual blocks, including the 1x1 shortcuts, are `QuantConv2d`. The stem convolution and the final linear layer stay in full precision. SGD with Nesterov momentum, coupled weight decay on weights with two or more dimensions, linear warm-up and cosine decay to `lr * min_lr_frac`, batch size 128, random crop with 4-pixel padding and horizontal flip done on the GPU, bf16 autocast. The whole training set is kept on the GPU.

After `--epochs` epochs, `--cooldown` further epochs run at the constant final learning rate. During the cool-down the level counts, flip counts and a latent EMA (beta = 1 - 2 / cool-down steps) are collected. At the end each deployment rule (snapshot, mode, dither-free, EMA-requantized, latent) is evaluated on the test set twice: with the training-time batch-norm statistics and after re-estimating them in one pass over 20000 training images without augmentation.

```bash
python cifar/train_cifar.py --alphabet ternary --seed 0 --out runs/cifar/ter_s0
python cifar/train_cifar.py --alphabet binary --seed 0 --out runs/cifar/bin_s0
python cifar/train_cifar.py --alphabet none --seed 0 --out runs/cifar/fp_s0
python cifar/train_cifar.py --dataset cifar100 --data ~/data/cifar100 --alphabet ternary --seed 0 --out runs/cifar100/ter_s0
```

| Option | Default | Meaning |
|---|---|---|
| `--alphabet` | `ternary` | As in `train.py` (`ternary`, `binary`, `none`, `uniform_k`, `levels:...`). |
| `--gamma` | `1.0` | Scale multiplier. |
| `--dz` | `0.5` | Ternary dead zone. |
| `--scale_rule` | `absmean` | `absmean`, `lsq` or `twn`. |
| `--epochs` | `160` | Epochs of the cosine schedule. |
| `--cooldown` | `10` | Cool-down epochs at the final learning rate. Keep it at 1 or more for quantized runs. The final duty-cycle evaluations need the counts collected there. |
| `--bs` | `128` | Batch size. |
| `--lr` | `0.1` | Peak learning rate. |
| `--min_lr_frac` | `0.01` | Final learning rate as a fraction of the peak. |
| `--wd` | `5e-4` | Weight decay. |
| `--momentum` | `0.9` | SGD momentum (Nesterov). |
| `--warmup_epochs` | `2` | Linear warm-up epochs. |
| `--seed` | `0` | Seed for initialisation, data order and augmentation. |
| `--data` | `~/data/cifar10` | Directory for the downloaded dataset. |
| `--dataset` | `cifar10` | `cifar10` or `cifar100`. |
| `--out` | required | Run directory. |
| `--device` | `cuda` | Device (`cuda` or `cpu`, with the GPU selected by `CUDA_VISIBLE_DEVICES`). |
| `--max_batches` | none | Limit batches per epoch (for debugging). |

Outputs in `--out`:

| File | Contents |
|---|---|
| `args.json` | Arguments. |
| `log.jsonl` | One record per epoch: `epoch`, `step`, `lr`, `train_loss`, `train_acc`, `test_acc`, `elapsed`, and for quantized runs `flip_mean`, `zero_mean`. |
| `final.json` | `test_snapshot_end_of_schedule` (with a cool-down), `n_params`, `n_quant`, `steps`, `elapsed`, `test_snapshot`, and for quantized runs `test_<rule>` and `test_<rule>_bnrecal` for `mode`, `dither_free`, `ema_requant`, `latent_fp` and `snapshot`, `per_layer` statistics (scale, zero fractions, flip rate, two-, three- and one-level fractions, mode/snapshot agreement, EMA/mode agreement, mean latent step, kurtosis, mean and standard deviation of the latent weights, latent density at the threshold and at zero) and their means. |
| `final_state.pt` | Per quantized layer `latent` (fp16), `counts`, `flipc`, `ema` (fp16), and `_other` with the remaining parameters. For `--alphabet none` a flat dict of all parameters. |

`train_cifar.py` does not write checkpoints. An interrupted run starts again from the beginning (delete the partial `log.jsonl`, which is opened in append mode).

### Batch-norm re-estimation on augmented data: `analysis/cifar_reeval.py`

Rebuilds the network from `final_state.pt`, forms the snapshot, mode, dither-free, EMA-requantized and latent weights from the stored counts and latents, and for each of them resets the batch-norm statistics, re-estimates them from `--batches` augmented training batches (random crop and flip) in train mode, and measures test accuracy and mean test loss.

```bash
python analysis/cifar_reeval.py runs/cifar/ter_s0 --batches 200
```

| Option | Default | Meaning |
|---|---|---|
| `run` | | CIFAR run directory (positional). |
| `--batches` | `200` | Training batches used for the re-estimation. |
| `--bs` | `128` | Batch size for the re-estimation. |
| `--device` | `cuda` | Device. |
| `--seed` | `123` | Seed for the batch order and augmentation. |

Output `reeval.json`: `batches`, `augmented`, and for every variant the accuracy (`snapshot`, `mode`, ...) and loss (`snapshot_loss`, `mode_loss`, ...). Full-precision runs give the snapshot only.

## Synthetic experiments

The scripts in `synth/` simulate many independent coordinates, each with a quadratic loss (h/2)(q - w*)^2 (h = 1) in its quantized weight q, a target w* drawn from a Laplacian with E|w*| = 1 (or a Gaussian with sigma = 1), and a latent w updated with the straight-through gradient h(q - w*). They need only NumPy or PyTorch and run on a CPU. `synth/e1.py` with its default size runs faster on a GPU.

### `synth/e1.py`

Runs 76 configurations and compares the measurements over the window after burn-in with closed-form values: the slope of the time-averaged q on w* (expected 1 for interior coordinates |w*| < s), snapshot penalty, flip rate, zero fraction, mode-rounded penalty, EMA-requantized penalty, fraction of interior coordinates that never flip, fraction visiting three levels, and curves binned by |w*|/s. The configurations are: a scale scan (Laplacian and Gaussian, SGD lr 1e-2 and Adam lr 1e-3, binary and ternary, s from 0.5 to 2.0 times E|w*|), a dead-zone variation (dz 0.25, 0.75), learning-rate variations, SGD with coupled weight decay 0.01 and 0.1, and AdamW with decoupled decay 0.1.

```bash
python synth/e1.py --out synth/results/e1 --device cuda:0
python synth/e1.py --out synth/results/e1 --device cpu --quick
```

| Option | Default | Meaning |
|---|---|---|
| `--out` | `results/e1` | Output directory (relative to the current directory). `e1_results.json` is rewritten after every configuration. |
| `--device` | `cuda:0` | Torch device. |
| `--n` | `1000000` | Coordinates. |
| `--steps` | `20000` | Steps. |
| `--burn` | `10000` | Burn-in steps before the measurement window. |
| `--quick` | off | Sets n = 200000, steps = 4000, burn = 2000. |

Use `--out synth/results/e1` if you also want `synth/adamw_slope_drop.py` to pick up the measured values.

### `synth/e1_noise.py`

Adds Gaussian gradient noise sigma * N(0, 1) with sigma in {0, 1, 10, 100} (Adam, lr 1e-3, s = 1, binary and ternary) and reports the slope of the time average on w*, the snapshot penalty against the noise-free closed form, flip rate (overall and in ten bins of |w*|/s) and zero fraction.

```bash
python synth/e1_noise.py --device cpu --n 200000 --steps 6000 --burn 3000 --seed 0 --out synth/results/e1_noise
```

Options: `--device` (`cpu`), `--n` (`200000`), `--steps` (`6000`), `--burn` (`3000`), `--seed` (`0`), `--out` (none. If given, `e1_noise.json` with the configuration, all runs and the closed-form penalties G_B(1), G_T(1) is written there).

### `synth/e1_noise_gamma.py`

For sigma in {0, 3, 10}, ternary and binary, and s in {0.75, 1.0, 1.25, 1.5, 2.0}: snapshot penalty, the gap between the snapshot penalty and the penalty of the average over the last 64 steps, mean Var(q)/s^2 over those 64 steps, and zero fraction. Prints only.

```bash
python synth/e1_noise_gamma.py --device cpu --n 100000 --steps 8000 --burn 4000
```

Options: `--device` (`cpu`), `--n` (`100000`), `--steps` (`8000`), `--burn` (`4000`, accepted but not used, since the window is always the last 64 steps).

### `synth/adam_relay_1d.py`

A single deterministic coordinate with Adam (lr 1e-3, betas 0.9 and 0.95, bias correction, no noise), binary and ternary, w* from 0.0 to 0.9 and several starting points. Reports the flip rate over the last 10000 of 20000 steps, the period of the steady-state level sequence (up to 2000 steps), flips per period, and the bias of the time-averaged q.

```bash
python synth/adam_relay_1d.py --out synth/results/adam_relay_1d.json
```

`--out` defaults to `synth/results/adam_relay_1d.json` relative to the current directory.

### `synth/adamw_slope_drop.py`

Computes the predicted drop of the regression slope of the time-averaged q on w* caused by decoupled AdamW weight decay 0.1 for ternary coordinates (and the slope under SGD with coupled decay 0.01 and 0.1), from moments of |w*| obtained by quadrature and, if PyTorch is available, from the same draw as `synth/e1.py`. If `synth/results/e1/e1_results.json` exists it adds the measured drop between the Adam and AdamW ternary runs at s = 1.

```bash
python synth/adamw_slope_drop.py --out synth/results/adamw_slope_drop.json
```

`--out` defaults to `results/adamw_slope_drop.json` inside the `synth/` directory.

### `synth/dither_lloyd.py`

Finds, by coordinate descent on a fine grid, the symmetric k-level quantizer that minimizes the dither functional (expected snapshot penalty when every weight dithers between its two neighbouring levels) and the one that minimizes the mean squared rounding error (Lloyd-Max), for k in {2, 3, 4, 5, 7, 15} and Laplacian and Gaussian weights. Prints the positive levels in units of E|w|, both functionals for both level sets and the relative gain in the dither functional.

```bash
python synth/dither_lloyd.py
```

### `synth/dlloyd_check.py`

Simulates STE with Adam (200000 coordinates, 20000 steps, Laplacian targets) with the 4- and 7-level quantizers found by `dither_lloyd.py` (levels are written in the script) and reports the measured snapshot and mode-rounded penalties next to the dither and MSE functionals of each level set.

```bash
python synth/dlloyd_check.py
```

### `synth/momentum_window.py`

Noise-free ternary coordinates with plain SGD and with heavy-ball momentum 0.9. Reports the RMS error of the duty-cycle average over windows of 16, 64, 256 and 1024 steps and the log-log slopes between them.

```bash
python synth/momentum_window.py
```

### `synth/window_rice_check.py`

Adam, ternary, s = 1, dz = 0.5, sigma in {0, 10}: (a) RMS error of the time average over windows of 16 to 4096 steps for interior coordinates, and (b) the level-crossing prediction of the flip rate, 2 (rho(+dz) + rho(-dz)) E|dw|, with the latent density and mean step measured in the simulation, against the measured flip rate.

```bash
python synth/window_rice_check.py
```

## Hardware, run time and interruptions

- The language-model trainer and the LM analysis tools need a CUDA GPU with bf16 support. Training state costs 16 bytes per parameter (fp32 weights, gradients and two AdamW moments) plus activations, which scale with `--micro_bs` and `--seq`. During the counting window each counted weight additionally needs 2 bytes per level for the counts (6 bytes for binary and ternary), 2 bytes for the flip count, and 8 bytes for the latent EMA and the previous latent copy, and `--lean` drops the last 8 bytes. The `XL16` model fits a 12 GB GPU with counting at micro-batch 2.
- Run time grows with the number of tokens and the model size. The trainer prints throughput in tokens per second every `10 * log_every` steps and stores it in `log.jsonl`.
- `train.py` writes `ckpt.pt` every `--ckpt_every` steps and resumes from it automatically when started again with the same `--out`. Checkpoints are not written inside the counting window, so an interruption there restarts the window from the last checkpoint before it. `log.jsonl` and `hist.jsonl` are opened in append mode, so records between the last checkpoint and the interruption appear twice after a resume.
- `analysis/probe.py` and `analysis/perbatch_eval.py` are resumable as described above, `analysis/zeroshot.py` saves results after every task, and `data/prepare.py` can restart from a given file and shard.
- The CIFAR trainer and the synthetic scripts do not checkpoint.

## Licence

MIT License, Copyright (c) 2026 Jyotiprakash Mishra. See `LICENSE`.
