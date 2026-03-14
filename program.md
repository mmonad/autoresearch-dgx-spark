# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create a worktree + branch**: Use `git worktree add ../autoresearch-<tag> -b autoresearch/<tag>` to create an isolated working copy. All experiment work happens in the worktree directory (`../autoresearch-<tag>`), NOT in the main repo checkout. This keeps the main repo clean and avoids conflicts if multiple experiments run in parallel.
3. **`cd` into the worktree**: All subsequent commands run from `../autoresearch-<tag>`.
4. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
5. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script supports two stopping modes, controlled by `TOKEN_BUDGET` in `train.py`:

- **Wall clock time** (default): `TOKEN_BUDGET = None` — training runs for the fixed `TIME_BUDGET` (5 minutes). Rewards both algorithmic quality and throughput. Good for practical optimization where speed matters.
- **Fixed token budget**: `TOKEN_BUDGET = 100_000_000` (or similar) — training runs for exactly N tokens regardless of speed. Isolates learning efficiency from throughput. Good for architecture research where you want to compare how well different models learn from the same data.

The user will tell you which mode to use. Launch with: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** But achieving this within a fixed wall clock budget is a dual challenge: you must both **train better** (algorithmic improvements — architecture, optimizer, hyperparameters) and **train faster** (throughput improvements — more tokens processed in the same 5 minutes). A change that improves per-step learning but halves throughput may be a net loss; conversely, a neutral-per-step change that doubles throughput gives you 2x the training steps, which can dramatically lower val_bpb. Always consider both dimensions.

**Hardware context — NVIDIA DGX Spark (GB10 Grace Blackwell Superchip):**

This system has a unique hardware profile that shapes what optimizations are effective:

| Spec | Value | Implication |
|---|---|---|
| GPU arch | Blackwell SM121a (sm_121a) | Not datacenter Blackwell (SM100). Some kernels/libraries don't support it. Only CUDA 13.0+ (cu130) has native codegen. |
| CUDA cores | 6,144 | ~1/15th of an H100. Compute is modest. |
| Tensor cores | 192 (5th-gen) | Support FP64/FP32/BF16/FP8/FP4. BF16 tensor peak ~125 TFLOPS. |
| Memory | 128 GB unified LPDDR5x | Shared coherently between CPU and GPU via NVLink-C2C. Huge capacity, but... |
| Memory bandwidth | 273 GB/s | **This is the primary bottleneck.** ~4.5x lower than an H100 (3.35 TB/s). The system is firmly memory-bandwidth-limited. |
| L2 cache | 24 MB (+ 16 MB L4 side cache) | Smaller than desktop GPUs. Cache-friendly access patterns matter more. |
| TDP | 240W (whole system) | Low power envelope means thermals are not an issue. |

**What this means for optimization strategy:**

- **You are memory-bandwidth-limited, not compute-limited.** The 273 GB/s bandwidth is the bottleneck, not the tensor core TFLOPS. Optimizations that reduce memory traffic (fewer parameters to load, better data reuse, fused operations) are more valuable than those that merely reduce FLOPs.
- **Throughput is king.** At ~140K tok/sec baseline and only ~91 steps in 5 minutes, every extra step matters. Changes that increase tok/sec (smaller models that still learn well, larger batch sizes that amortize overhead, fewer memory-bound operations) directly translate to more training and lower val_bpb.
- **Kernel/software support is limited.** SM121a is not mainstream. Flash Attention 3 doesn't work (we use flex_attention via Triton). torch.compile with CUDA graphs can crash. Some Triton kernel options cause failures. Always be cautious with advanced kernel tricks — they may silently fail or produce wrong results on this arch.
- **FP8 training is theoretically supported** by the 5th-gen tensor cores but driver/compiler maturity for FP8 on SM121a is uncertain. If attempting lower-precision training, validate carefully.
- **Unified memory means no CPU-GPU transfer cost** but also means CPU and GPU compete for the same 273 GB/s bandwidth. CPU-heavy data preprocessing during training could steal bandwidth from the GPU.
- **Large model capacity (128 GB) is a trap.** You can fit very large models in memory, but the bandwidth bottleneck means you can't feed them fast enough. The sweet spot is a model that's large enough to learn well but small enough to achieve high throughput given the bandwidth constraint.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs in a dedicated worktree + branch (e.g. worktree `../autoresearch-mar5` on branch `autoresearch/mar5`). All work happens in the worktree directory, not the main repo checkout.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Beyond tuning: when to shift gears

Hyperparameter sweeps have diminishing returns. When most experiments are discards and improvements are within noise, you've reached a plateau — the model is locally optimal under its current architecture. **This is the signal to stop tuning and start innovating.**

At this point, single-parameter changes won't move the needle. What will:

**Use `results.tsv` as a research journal.** Before proposing a new experiment, re-read the full results log. It tells you what's been tried, what worked, what failed, and why. Use it to avoid repeating past mistakes and to spot patterns — e.g. "every throughput-reducing change was a net loss" or "optimizer changes near X value were all noise." Let the data guide your next hypothesis.

**Search for new ideas.** Use web search to find recent papers, blog posts, and techniques. Look for what's working in efficient language model training *right now* — the field moves fast. Don't limit yourself to what you already know. Read broadly: new optimizers, new architectures, new training recipes, new attention mechanisms. Implement what looks promising.

**Make structural changes, not incremental ones.** Rewrite the model architecture. Change how attention works. Replace components entirely. Try ideas from adjacent fields. The expected value of bold experiments is higher than safe ones when you're on a plateau — a risky rewrite that crashes twice but eventually gives -0.01 is worth more than twenty safe tweaks at -0.0002 each.

**Combine near-misses.** If several changes were each slightly worse individually, try combining them — interactions between changes can be non-linear. Two changes that each hurt throughput slightly might together enable a qualitatively different operating point.

**Revisit assumptions.** Question everything inherited from earlier experiments: Is this the right model size? The right depth/width ratio? The right optimizer? The right way to do attention? Earlier experiments established these under different conditions — they may no longer be optimal.
