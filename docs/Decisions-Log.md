# Decisions log

One entry per decision. Date, what was decided, why, and what would make you revisit it. This file becomes the "methodology" and "discussion" chapters later, so write the reasons, not just the outcome.

---

## 2026-09-17: Project name and framing

**Decided:** Project is called Anneal. Framed as "how much does targeted fine-tuning improve an open-weight model's coding ability on consumer unified-memory hardware," not "build a new fine-tuning method."

**Why:** The contribution is a rigorous, reproducible measurement on a platform with almost no documented practice, which is realistic in a year and honest about the hardware.

**Revisit if:** A supervisor pushes for a methodological contribution. Even then, the measurement is still the foundation.

---

## 2026-09-17: Scope, from the project agreement

**Decided:** One base model, 20B to 40B parameters, fine-tuned on the owned 64GB machine. The comparison is that model before and after fine-tuning. No second model in any role.

**Why:** It is the agreement. It is also the cleanest experimental design: one variable (the fine-tune), everything else held constant.

**Revisit if:** Never.

---

## 2026-09-18: Hardware confirmed

**Decided:** GMKtec EVO-X2, 64GB unified memory, 1TB NVMe. Sole compute for the project.

**Why:** Owned outright, no departmental dependency. 64GB was the configuration within budget in the UK.

**Revisit if:** Storage becomes the bottleneck (external NVMe).

---

## 2026-09-19: Compute constraint, local box only

**Decided:** Every training run, evaluation and inference in this project happens on the owned EVO-X2. No rented GPUs, no cloud instances, no departmental cluster, at any stage, including the write-up.

**Why:** It is part of the project agreement and the brief's resource statement ("no departmental hardware or compute access should be needed"). It also keeps the result honest: the whole point is what consumer hardware can do on its own.

**Consequences:** any model or precision that does not fit in 64GB is simply out of scope, not a reason to look elsewhere. Anything that fails to fit gets measured and written up as a finding (`scripts/measure_memory.py`), then dropped.

**Revisit if:** Never.

---

## 2026-09-19: Base model = Qwen3.8-27B, trained in 8-bit LoRA

**Decided:** The one model is `Qwen/Qwen3.8-27B` (dense, Apache-2.0, August 2026). Training is 8-bit LoRA by default, 4-bit QLoRA as the fallback. Baseline and fine-tuned evaluations run at the same quantisation and thinking mode.

**Why:** It is in range, dense (so quantised training actually shrinks it), on Unsloth's supported path for the family, and among the strongest open 27B coders on the model card's numbers. 8-bit over 4-bit because Unsloth warns 4-bit is lossy on this family.

**Memory evidence** (published; measure on day one): bf16 weights are 55.6GB on disk. Unsloth's requirements table lists a 27B at 64GB for 16-bit LoRA and 22GB for 4-bit QLoRA as "absolute minimums"; its Qwen3.8 page lists 8-bit weights at 31GB and bf16 at 56GB. Usable memory here is 48GB (Windows) or ~56GB (Linux). bf16 LoRA is over by 8–16GB even on Linux; 8-bit lands around 37–41GB; 4-bit around 22GB. Details in `Hardware-Notes.md`.

**Withdrawn on the way here:** an MoE model (35B, ~3B active) was considered first, both as the training target and later as an inference-only reference. As a training target it cannot fit: bitsandbytes leaves fused expert tensors in bf16, so it loads at ~66GB at any bit width. As a reference it violates the one-model agreement. A 9B dense model was also considered and is below the agreed range. All three are out.

**Revisit if:** Qwen3.8-27B fails on ROCm in a way that a version bump does not fix; then `Model-Candidates.md` lists same-shape and other dense fallbacks. A switch restarts the baselines and is still one model.

---

## 2026-09-19: Context length

**Decided:** Inference and every evaluation run at the model's full native context, 262,144 tokens, with the KV cache quantised to q8 (8GB at full context; 16GB in f16 also fits). Training examples are 4,096 tokens (8,192 in 4-bit if the day-one measurement allows), which does not change the model's context window.

**Why:** The context is affordable here because only 16 of 64 layers are full attention, so the KV cache is 64KB per token. Running at full context makes repository-scale tasks possible in the personal evaluation set, which is the use a local assistant is for. Training at long sequence lengths is bounded by activation memory (an 8k example needs ~54GB even in 4-bit), and it is unnecessary: LoRA does not alter positional handling.

**Consequences:** the personal task set gains repository-scale tasks; the forgetting check gains a long-context probe so a short-example fine-tune cannot silently dull long-context behaviour; prompt-processing time at full context is measured on day one and budgeted for.

**Revisit if:** the long-context probe drops after fine-tuning (then mix in longer examples), or prefill at full context proves slow enough to make repository-scale evaluation impractical (then cap the personal tasks at 64k–128k).

---

## PENDING: Operating system for the EVO-X2

The 64GB is one physical pool on either OS. Windows splits it into a dedicated GPU pool (max 48GB on this unit) plus shared memory the GPU can use for inference at a speed penalty; whether training can use the shared part is unverified. Linux has no split: AMD's guidance is a 0.5GB carve-out and 52–56GB for the GPU through GTT at full speed. Thinking-mode training at seq 4096 in 8-bit estimates at ~49GB, over the Windows dedicated pool and inside the Linux budget. Leaning Linux for that reason; it is also where the tooling and AMD's documentation live. If Windows is tried first, `measure_memory.py` at 8-bit / 4096 settles the open question in an afternoon. Decide before the machine arrives.

---

## 2026-09-19: Thinking mode ON

**Decided:** The model is trained and evaluated with thinking enabled everywhere: training rows carry `<think>` traces, the baseline and every fine-tune are scored with thinking on, at Qwen's recommended sampling (temperature 0.6, top-p 0.95) with fixed seeds.

**Why:** It is the mode the model was RL-tuned in and the mode its published scores use. Training against it would most likely lower results.

**Consequences:**
- Training data must contain reasoning traces: `nvidia/OpenCodeReasoning`, or traces generated by the base model itself with `scripts/generate_traces.py` (optionally test-filtered). Plain instruction data such as Magicoder cannot be used as-is.
- Sequence length 4096. Estimator: 8-bit ~49GB (Linux only, tight), 4-bit ~35GB (fits everywhere). Measure on day one and pick; this pushes the OS decision toward Linux.
- Reasoning budget 32,768 tokens (decided 2026-09-19), total generation budget 40,960 so the answer always has room. Identical for baseline and every fine-tune.
- Evaluation time is set by trace length, which is measured by the first 20 problems (`LIMIT=20 run_eval.sh`), not assumed. Measured speeds on this chip: Q8 7.2 tok/s plain, 15.3 with MTP speculative decoding; Q4_K_M 12.3 plain, ~24.5 with MTP (sources in `Hardware-Notes.md`). At 2k tokens per problem that is ~20 h for a full pass at Q8 with MTP and ~14 h at Q4 with MTP. One seed while iterating, three seeds for the headline numbers.
- Greedy decoding is out (Qwen advises against it with thinking on); sampling adds variance, hence the seeds. Sampling also lowers MTP acceptance, which is already reflected in the measured figures above.

**Revisit if:** Never for the mode. The data source is still open below.

---

## PENDING: Training dataset

Must contain reasoning traces (see the thinking-mode entry). Options: `nvidia/OpenCodeReasoning` (ready-made, CC-BY-4.0, verify on the card), self-generated traces from the base model via `scripts/generate_traces.py`, or a hybrid. Leaning hybrid: a few hundred self-generated, test-filtered rows on the task types the personal test set covers, plus a length-filtered slice of OpenCodeReasoning for volume. See `Dataset-Candidates.md`.

---

<!-- Template for new entries:

## YYYY-MM-DD: Title

**Decided:**

**Why:**

**Revisit if:**

-->
