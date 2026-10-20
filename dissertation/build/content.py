# Content for the Anneal dissertation outline.
# Inline markup: **bold**, *italic*, [@key] citation, {{text}} yellow placeholder, `code`.

TITLE = "Anneal: Memory-Constrained LoRA Fine-Tuning of Qwen3.8-27B for Software Engineering at 256k Context"
AUTHOR = "Callum Challinor"
STUDENT_ID = "3308299"
SUPERVISOR = "Supervisor: Yuanlin Gu"
DATE = "October 2026"
AI_NOTE = "I used Claude as an AI assistant to monitor training and to help with document creation."

BLOCKS = [
("h1", "Introduction"),
("bf", "Large language models (LLMs) are widely used to write and repair code, but the most capable run as paid cloud services, so source code leaves the developer's machine. Open-weight models run locally, yet adapting one to a particular kind of work normally needs datacentre hardware. This project, Anneal, investigates whether a low-rank adapter (LoRA) trained only on public datasets can measurably improve the software-development ability of the open-weight Qwen3.8-27B model, when training must fit within the 32 GB of memory allocated to the GPU of one AMD machine and the adapted model must keep its 256k-token context window, and how that gain changes as a months-long improvement loop adds data."),

("h2", "Background and Context"),
("bf", "Programming assistants now carry out whole tasks: implementing a function from a description, fixing a bug across several files, or answering questions about an unfamiliar codebase. The last depends on context length: a 256k-token window holds tens of thousands of lines, enough for a medium-sized repository in one request."),
("bt", "Running such a model locally matters to anyone who cannot send code to a third party: companies with proprietary code, developers working offline, and students without a cloud budget. Strong open-weight models make this possible but are general-purpose, and full fine-tuning of a 27-billion-parameter model needs far more memory than any desktop offers. Parameter-efficient fine-tuning trains a small set of extra parameters instead, bringing the memory needed within reach of a high-end workstation. Whether this improves software engineering in a modern 27B model without losing long-context ability has not been established; this project measures it."),

("h2", "Problem Analysis"),
("bf", "The problem is defined by four constraints, each of which becomes a requirement."),
("bt", "**R1 Memory.** In 16-bit precision the weights of Qwen3.8-27B alone occupy about 55 GB, more than the 32 GB the training machine allocates to its GPU, which rules out full fine-tuning and 16-bit LoRA. Anneal quantises the frozen base to 4 bits, as in QLoRA [@qlora], leaving most of the budget for the adapter, optimiser state and activations."),
("bt", "**R2 Context.** The model supports 262,144 tokens natively; only one layer in four uses full attention, and the rest are Gated DeltaNet linear-attention layers [@qwen]. Activation memory limits training examples to about 8,192 tokens, and short-example fine-tuning risks dulling long-range behaviour unnoticed [@longlora], [@ruler], so retention must be measured, not assumed."),
("bt", "**R3 Data.** The model reasons in a `<think>` block before answering, so every training row needs a trace, or the model learns to skip reasoning. Data must come from public datasets licensed for training and must not overlap the evaluation sets, or gains could reflect memorisation."),
("bt", "**R4 Deployment.** The model runs locally as a quantised GGUF file through llama.cpp, so the base and adapted models are evaluated identically in that form."),
("bt", "Three scenarios drive the evaluation: (S1) writing a function from a specification; (S2) changing code across a real repository; and (S3) reasoning about a large repository held in the context. Professional issues are licensing, reproducibility, honest reporting of negative results, and the time and energy each run costs, which will be logged."),

("h2", "Scope and Objectives"),
("bf", "In scope are one base model, the post-trained Qwen3.8-27B checkpoint (no pre-trained base is released [@qwen]), LoRA over a 4-bit base, thinking mode throughout, and one owned machine with no cloud or departmental compute. Out of scope are full fine-tuning, a second model, context beyond 256k, the vision encoder, reinforcement learning and user studies. The objectives are:"),
("obj", "O1", "Build a reproducible pipeline on PyTorch and ROCm that trains a Qwen3.8-27B adapter on examples of up to 8,192 tokens within the machine's 32 GB GPU allocation and survives interruptions to multi-week runs, by the end of Autumn week 12."),
("obj", "O2", "Assemble licence-checked public training sets with reasoning traces, decontaminated against every gate and evaluation set and documented in a data card, before each set's first loop round."),
("obj", "O3", "Produce an adapter that raises pass@1 on HumanEval+ and MBPP+ over the base model, with a 95% confidence interval for the difference that excludes zero, and report the same comparison on post-release LiveCodeBench problems and a personal task set."),
("obj", "O4", "Keep long-context accuracy (RULER and LongCodeBench, 32k to 256k tokens) and a non-coding check within 3 percentage points of the base model."),
("obj", "O5", "Measure how coding gain and retention change as loop rounds add public data, and the effect of rank, target modules, example length, data source and base precision through ablations."),
("obj", "O6", "Publish the pipeline, configurations, data card and adapter so that the results can be reproduced."),
("bt", "The expected results are an adapter, an open pipeline for AMD hardware and a measured account of what LoRA can achieve under this budget, including negative results. This gives evidence on parameter-efficient fine-tuning outside the NVIDIA CUDA ecosystem in which LoRA and QLoRA were developed [@lora], [@qlora]."),

("h1", "State of The Art"),
("bf", "Three bodies of work shape this project: parameter-efficient fine-tuning, which makes training possible within 32 GB; long-context research, which defines what must be preserved; and execution-based code benchmarks, which define how improvement is measured. Six papers are reviewed below, each with what Anneal takes from it, followed by the technologies the project builds on."),

("h2", "Parameter-Efficient Fine-Tuning"),
("h3", "Low-Rank Adaptation"),
("bf", "Hu et al. [@lora] freeze the pretrained weights and learn each weight update as the product of two small matrices of rank r. On GPT-3 175B this cut the number of trainable parameters 10,000-fold and GPU memory threefold while matching full fine-tuning, and a merged adapter adds no inference latency. Its weaknesses for this project are that the frozen base stays in 16-bit precision, about 55 GB at 27B, and that the evaluation used short natural-language tasks and adapted only the attention projections. Anneal adopts the mechanism and separate adapter loading, but treats rank and target modules as variables to test rather than inheriting the original defaults."),
("h3", "Quantised LoRA"),
("bf", "Dettmers et al. [@qlora] backpropagate through a frozen base quantised to 4-bit NormalFloat (NF4), adding double quantisation and paged optimisers. They fine-tuned a 65B model on one 48 GB GPU while matching 16-bit fine-tuning, and found that LoRA on all linear layers, not only attention, was needed to reach that quality. This is the strongest evidence that Anneal's budget is workable, since a 27B base leaves more room for activations. However, the evaluation centred on chat and knowledge benchmarks with short sequences on NVIDIA GPUs, and the training format differs from the GGUF format used at deployment. Anneal adopts the recipe, a 4-bit NF4 base with LoRA on attention and feed-forward projections, and evaluates in the deployment format. Because Unsloth advises against 4-bit training on the closely related Qwen3.5 models, citing larger-than-normal quantisation differences [@unslothq35], Anneal starts from the pre-quantised checkpoint Unsloth's guide uses for QLoRA [@unslothq38], and ablation A6 trains one 8-bit adapter to measure the remaining cost."),
("h3", "What LoRA Learns and Forgets"),
("bf", "Biderman et al. [@biderman] compared LoRA with full fine-tuning on code and mathematics. At standard ranks LoRA learned substantially less, but better preserved performance outside the target domain, and full fine-tuning learned weight changes of 10 to 100 times higher rank. They also report that forgetting grows with training duration, a direct risk for a months-long loop. The study used Llama-2-7B, far weaker than Qwen3.8-27B, so whether a strong base leaves similar room to improve is open. For Anneal the paper sets realistic expectations and justifies the rank ablation and the non-coding check; its finding that LoRA forgets less is also the main reason to expect long-context ability to survive short-example training."),

("h2", "Long-Context Behaviour"),
("h3", "Long-Context Fine-Tuning"),
("bf", "Chen et al. [@longlora] extend context windows using shifted sparse attention to cut training cost. They show that plain LoRA is not enough for extension: on Llama 2 7B at 32k tokens it reached a perplexity of 11.44 against 8.08 for full fine-tuning, while LoRA with trainable embedding and normalisation layers reached 8.12. Their largest run used eight A100 GPUs. Anneal preserves an existing 256k window rather than extending one, and shifted sparse attention assumes standard attention in every layer, whereas three in four Qwen3.8 layers use DeltaNet, so the method does not transfer. Two lessons do: adapters can affect long-range behaviour in non-obvious ways, so retention must be tested, and if it fails, long examples or trainable normalisation layers are evidence-based remedies (ablation A5)."),
("h3", "Measuring Effective Context"),
("bf", "Hsieh et al. [@ruler] show that the popular needle-in-a-haystack test overstates context ability. RULER adds multi-key retrieval, variable tracking, aggregation and question answering at configurable lengths. Almost every model tested passed the simple needle test, but only about half of those claiming 32k tokens or more performed satisfactorily at 32k. RULER is synthetic and contains no code, its main weakness here, so Anneal runs it at 32k to 256k alongside LongCodeBench (Table 1), which covers scenario S3."),

("h2", "Evaluating Software-Engineering Ability"),
("bf", "Jimenez et al. [@swebench] built SWE-bench from 2,294 GitHub issues across 12 Python repositories: a model must edit the repository until the project's own tests pass, which matches scenario S2 far better than single-function benchmarks. At publication the best model resolved 1.96% of issues. Its weaknesses are cost and noise. Each task needs a container and a token-hungry agent loop, scores depend on the agent scaffold, and underspecified tasks led to SWE-bench Verified, a human-validated subset of 500 [@verified]. Public repositories also risk contamination. At this chip's generation speed (Section 3.5) even a subset would take days per system, so Anneal instead borrows SWE-bench's execution-based design for a personal set of repository-scale tasks drawn from the author's own code, which no public model has seen."),

("h2", "Technologies and Related Products"),
("bf", "Table 1 summarises the alternatives considered and why each choice was made."),
("table", "Table 1. Technology choices.", [1700, 2600, 3996],
  ["Component", "Options considered", "Choice and reason"],
  [
   ["Training stack", "Unsloth; PEFT with bitsandbytes; CUDA-only tools", "Unsloth [@unsloth] on PyTorch with ROCm [@rocm], running natively on Windows 11, with the pre-quantised 4-bit checkpoint (bitsandbytes [@bnb]) that Unsloth documents for Qwen3.8-27B [@unslothq38]. Plain PEFT [@peft] is the backup; CUDA-only tools do not run here."],
   ["Inference", "llama.cpp (GGUF); Ollama; PyTorch", "llama.cpp [@llamacpp], Vulkan build: full 262k context with a quantised KV cache, separate reasoning output and runtime adapters."],
   ["Code benchmarks", "HumanEval+ and MBPP+; LiveCodeBench; SWE-bench Verified", "EvalPlus [@evalplus] HumanEval+ and MBPP+ (542 problems, far more tests than the originals) as the primary measure; LiveCodeBench [@lcb] problems released after the model guard against contamination; SWE-bench is too slow (Section 2.3)."],
   ["Long-context code", "RepoQA; LongCodeBench", "LongCodeBench [@longcode], which covers 32k to 1M tokens; RepoQA [@repoqa] defaults to 16k-token contexts."],
   ["Training data", "Public AI-written traces; human-written sets; own traces; teacher API; SWE-Gym", "DeepSeek-V4-Pro code traces [@tachibana], [@banaxi] (Apache-2.0) and AoPS maths [@nemomath] (CC-BY-4.0). Own and OpenCodeReasoning [@ocr] traces only in the first run; human-written sets lack traces; public agent-session steps [@agentset] have redacted tool calls; API-generated rows are not public; SWE-Gym [@swegym] trajectories exceed 8,192 tokens."],
   ["Existing products", "Cloud assistants (e.g. GitHub Copilot); vendor code models", "Cloud assistants send code off the machine; vendor models cannot be specialised further without solving the problem Anneal addresses."],
  ]),
("bt", "In summary, QLoRA fixes the training configuration; LoRA and Biderman et al. frame what improvement to expect and which hyperparameters to test; LongLoRA and RULER define the long-context risk and how to measure it; and SWE-bench shapes the realistic tasks. Anneal's contribution is to combine these at 256k context, on AMD hardware, within 32 GB, and to report the outcome openly."),

("h1", "Prototype Development"),
("bf", "The prototype is a version-controlled repository of scripts and configuration for every stage in Figure 1. Drafted before access to the training machine, it was brought up there in October 2026; the first real run, round 0 of the loop (Section 3.2), started on 19 October, with scores due on 21 October."),
("fig", "fig1.png", "Anneal pipeline. Baseline (B) and adapted models (A) share one quantised base."),

("h2", "Memory: Estimate, Then Measure"),
("bf", "`memory_estimate.py` estimates training memory from the model's shape, and `measure_memory.py` replaces the estimate by running training steps and recording the peak. The estimator models plain Hugging Face training, but `train_longrun.py` uses Unsloth [@unsloth], which changes two large terms (Table 2). Qwen3.8's 248,320-token vocabulary makes the standard loss expensive; Cut Cross-Entropy, which Unsloth adopts, cut Gemma 2's classifier-head training memory from 28 GB to 1 GB [@cce], shrinking this model's loss from about 12.6 GiB to 0.5 GiB."),
("bt", "Table 2 decided the precision. An 8-bit base fits only if Unsloth's optimisations work on this chip, whereas a 4-bit base fits on either path; ablation A6 measures its cost in quantisation error. Measured on the machine at 8,192 tokens with a rank-16 adapter, the peak was 25.2 GiB, inside the 32 GB allocation."),
("table", "Table 2. GPU memory per step at rank 16: estimated at 4,096 tokens, measured at 8,192 (GiB).", [4296, 2000, 2000],
  ["Component", "4-bit (chosen)", "8-bit (A6)"],
  [
   ["Frozen linear layers", "11.7", "22.7"],
   ["Embeddings and output layer", "4.7", "4.7"],
   ["LoRA, gradients, optimiser", "0.7", "0.7"],
   ["Checkpoints (in host memory)", "0 (2.5)", "0 (2.5)"],
   ["Recomputed activations", "1.0", "1.0"],
   ["Loss (fused)", "0.5", "0.5"],
   ["Runtime allowance", "2.0", "2.0"],
   ["**Total, Unsloth**", "**about 21**", "**about 32**"],
   ["Total, plain PEFT", "40 to 46", "51 to 57"],
   ["Measured peak at 8,192 tokens", "25.2", "not run"],
  ]),

("h2", "Data Pipeline"),
("bf", "From round 1, the loop trains only on public datasets, never on the base model's own rows, which would reinforce its biases. No large public set of test-verified traces from an equally strong teacher exists, so loop rows are untested traces from DeepSeek-V4-Pro, level with Qwen3.8-27B on LiveCodeBench (89.8 against 90.3 [@dsv4], [@qwen]): 85% Tachibana4 code traces [@tachibana], 5% a second V4-Pro code set [@banaxi] and 10% AoPS maths [@nemomath]. A public agent-session set was dropped: 96% of its steps redact tool calls. `prepare_dataset.py` keeps rows with a `<think>` trace, drops rows over 8,192 tokens and removes near-duplicates of evaluation prompts (13-gram or Jaccard overlap). An embedding search on the otherwise idle NPU also flags reworded copies, which n-gram checks miss [@yang]; it flagged 33 of 15,276 loop rows without measurably slowing training."),
("bt", "The first run, round 0, is a comparison run using the model's own rows: 500 base-model answers to OpenCodeInstruct [@oci] prompts that passed their tests, as in ReST-EM [@restem], plus 1,500 OpenCodeReasoning rows, whose DeepSeek-R1 teacher scores only 63.5 on LiveCodeBench [@deepseek]. The base model passed 54% of attempts, but traces hitting the 6,144-token budget (16% of attempts) were rejected, so kept traces are short (median 436 tokens). Model-written code runs in a Windows AppContainer sandbox without network, file or administrator access, which passed 264 escape checks; 97.5% of reference solutions pass inside it."),

("h2", "Training"),
("bf", "Training runs as an unattended keep-if-better loop. Round 1 starts from the base model, never the first-run adapter; each later round continues the best adapter on 1,000 new public rows plus 25% replayed from its earlier rows, re-warming and decaying the learning rate, which in continual pre-training matched retraining on all data [@ibrahim]. Each round is scored on a held-out gate, never trained on or reported: 195 decontaminated CodeContests [@alphacode] validation and test problems with a passing reference, plus 300 OpenCodeInstruct problems. A round is kept only if a one-sided exact sign test against the best adapter gives p ≤ 0.05, its CodeContests part drops at most 5 points, 100 MMLU-Pro [@mmlupro] questions outside computing stay within 5 points of the base, and sample answers are clean. A kept adapter is re-scored with a fresh seed that becomes the new reference. A rejected round's adapter is deleted, its scores kept and its rows rolled into the next attempt, up to 3,000 new rows, because a real 2-point gain passes the gate only 24 to 44% of the time (power calculation). When fresh rows run out, a two-pass round runs before the loop waits for new sets. EvalPlus, scored only on kept adapters, never selects them."),
("bt", "`train_longrun.py` uses a warmup-stable-decay schedule [@wsd] (Listing 1). To survive crashes and updates over months, it saves every 10 steps, resumes when the same command is re-run, and refuses to resume if the training file has changed. In a CPU test with a tiny model of the same architecture, a run killed with `kill -9` resumed to bit-identical losses and weights."),
("code", "Listing 1. Excerpt of the first-run configuration.", [
 "base_model: unsloth/Qwen3.8-27B-unsloth-bnb-4bit",
 "seq_len: 8192                  # per example, not the context",
 "lora_r: 16                     # 64 in a later loop round",
 "target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,",
 "                 down_proj]    # + DeltaNet at rank 64",
 "schedule: wsd                  # 20 warmup steps, last 20% decays",
]),
("bt", "Small test fixtures caught four silent bugs, all fixed: the unit-test filter passed every answer, length was checked in characters so over-length rows could be truncated, the targets missed the 48 DeltaNet layers' projections, and a seed set after adapter creation made identical runs diverge from step 3."),

("h2", "Bring-up and First Run"),
("bf", "Bring-up and the first run raised the issues in Table 3. The run trained on 7.39 million tokens at 99.5 tokens per second; validation loss fell from 0.124 to 0.050."),
("table", "Table 3. Issues found during bring-up and the first run.", [2200, 2900, 3196],
  ["Issue", "Cause", "Resolution"],
  [
   ["Base model looked broken in training (perplexity 56 to 2,100)", "The loss scored the template's system prompt; a weight-order patch written to fix it was wrong", "Parity test on identical tokens: unpatched transformers matched llama.cpp (loss 0.515 vs 0.519, 95% top-1 agreement); patch removed, loss on answers only"],
   ["DeltaNet convolution slow", "The fast kernel needs an administrator build; PyTorch's fallback is slow here", "Shift-and-add rewrite, checked against a float64 reference: 61 to 97 tokens per second"],
   ["Training aborted on a long row that followed shorter ones", "Shared GPU memory grows to the longest row seen and is never returned, starving the fused loss", "Fixed loss chunk size and a warm-up step on the longest row: peak 28.1 GiB dedicated plus 8.3 shared, losses unchanged"],
   ["EvalPlus scorer does not run on Windows", "It relies on Unix-only signals and resource limits", "Sandboxed port; reference solutions score 378/378 (MBPP+) and 163/164 (HumanEval+)"],
   ["llama-server exited whenever an adapter was loaded", "Vulkan cooperative-matrix kernels fail with adapters on batches over 8 tokens", "Kernels off for every model: decoding unchanged, prompts 2.3 times slower"],
  ]),

("h2", "Export and Evaluation"),
("bf", "The evaluation script serves the model through llama.cpp at the full 262k context with thinking on and scores HumanEval+ and MBPP+ with the Windows EvalPlus port. HumanEval+ is also reported without HumanEval/32, which fails every solution through an EvalPlus 0.3.1 bug, and, for round 0, without three problems flagged in its own rows. A 20-problem calibration chose four parallel requests without multi-token prediction (MTP): 29.2 tokens per second in total, with answers averaging 877 tokens, so one 542-problem pass takes about 4.6 hours. Because parallel output drifts at near-ties, every model uses this one setup."),
("bt", "The adapter is converted alone to GGUF and loaded over the same base, which avoids a 55 GB merged model and keeps the base weights identical. Trained against NF4 but served over Q4_K_M, the smoke-test adapter gave losses of 0.756 in training and 0.728 in llama.cpp on 779 identical answer tokens, with 96% top-1 agreement, as close as the base model alone, so evaluation measures the adapter as deployed."),

("h2", "Critical Assessment"),
("bf", "The measurement-first design has paid off: the parity test caught a wrong fix before any real run used it. Speed and memory were measured only at rank 16, and the first run's scores will be the first evidence that training helps. The Windows workarounds (a bundled C compiler, the convolution rewrite, disabled Vulkan kernels) could break on an update, so versions are pinned. With 20 to 50 personal tasks, confidence intervals will be wide. Only 4,561 of Tachibana4's 17,234 rows pass the filters at 8,192 tokens, enough for about five rounds until 16,384-token training is measured."),

("h1", "Evaluation Plan"),
("bf", "Each objective has a success criterion (Table 4). The base (B) and adapted (A) models run on the same Q4_K_M GGUF with identical settings and seeds. The model card recommends temperature 1.0 and a longer reasoning allowance for thinking mode [@qwen]; Anneal keeps 0.6 and 32,768 tokens to bound evaluation time, so absolute scores may sit below the card's."),
("table", "Table 4. Objectives, measures and success criteria.", [800, 4300, 3196],
  ["Obj.", "Measure", "Success criterion"],
  [
   ["O1", "Peak GPU memory per run; resume after a forced stop", "Within the 32 GB GPU allocation at 8,192 tokens; resumed run matches"],
   ["O2", "Data card; overlap checks against the gate and every evaluation set", "Every licence recorded; no detected overlap"],
   ["O3", "HumanEval+ and MBPP+ pass@1, mean of three seeds (primary); post-August-2026 LiveCodeBench; 20 to 50 personal tasks", "A beats B on HumanEval+ and MBPP+ with a 95% CI excluding zero; the others reported with CIs"],
   ["O4", "RULER and LongCodeBench at 32k to 256k and 100 MMLU-Pro questions, for each kept adapter", "A within 3 percentage points of B on each"],
   ["O5", "Gate scores of every round; EvalPlus (one seed) for each kept adapter; ablations and the paired round", "Curve and each effect reported with 95% CIs"],
   ["O6", "Clean re-run from the published configuration", "Main result reproduced within its CI"],
  ]),
("bt", "**Statistics.** Pass rates are averaged over seeds; 95% confidence intervals for A minus B come from a paired bootstrap over problems, with a paired permutation test for significance. In calibration, 2 of 20 answers changed between two runs of the same model, so one seed over 542 problems resolves only a gain of roughly 3 points (an estimate); three seeds narrow this."),
("bt", "**Data scaling.** Published results suggest small, flattening gains: supervised fine-tuning gains are log-linear in data and smaller for stronger models [@yuan], and LoRA learns less than full fine-tuning and forgets more the longer it trains [@biderman]. Round 0, three-quarters from a weaker teacher, may show no clear gain, and many rounds may be rejected; a flat curve is a result, not a failure, and shows how much data is worth the machine time."),
("bt", "**Ablations.** One loop round tests A1 rank and A2 DeltaNet targets together: rank 64 on every projection (435M trainable parameters against 80M), retrained from the base on all kept rows. A paired round retrains a kept adapter's rows from the base, comparing continuing with replay against retraining. Separate 1,000-row runs cover A3 example length (4,096, 8,192), A4 OpenCodeReasoning alone against round 0's mix, A5 long examples or trainable normalisation if a kept adapter fails O4, and A6 an 8-bit base."),
("bt", "**Contamination.** LiveCodeBench is limited to problems released after the model (August 2026 [@qwen]). Two code sets are undated, so training rows are also checked against these problems."),

("h1", "Project Plan"),
("bf", "The first real run has trained (Section 3.4), and everything else depends on its scores. Of about 136 usable machine days from November to March, roughly 15 go to ablations and 20 to evaluation; loop rounds use the rest, since only one job can use the GPU at a time."),
("plan", "autumn", [
   ["Autumn: 7 - 8", "Round 0 (500 own and 1,500 OpenCodeReasoning rows) trained and scored against the base. Build the gate; start the loop.", "First before-and-after scores"],
   ["Autumn: 9 - 10", "Three-seed baseline; write the personal tasks. Ablations A3 and A4, pausing the loop.", "Baseline results with spread"],
   ["Autumn: 11 - 12", "Remaining ablations, including A6. Loop continues.", "Ablation results; working pipeline (O1)"],
   ["Autumn: 13 - 15", "Exams. The loop runs unattended.", ""],
  ]),
("plan", "spring", [
   ["Winter break", "Loop continues; ablation winners restart it from the base.", "Data card (O2)"],
   ["Spring: 1 - 2", "Loop continues. Interim demo. EvalPlus on new kept adapters.", "Scaling points"],
   ["Spring: 3 - 4", "Loop continues. Long-context checks on the best adapter.", "Retention results (O4)"],
   ["Spring: 5 - 6", "Loop continues. Poster session.", "Scaling points; poster"],
   ["Spring: 7 - 8", "Freeze the best kept adapter. Three-seed evaluation.", "Scaling curve and final results (O3 to O5)"],
   ["Spring: 9 - 10", "Reproducibility package. Draft method and results; send to supervisor.", "Public repository (O6); chapters 4 and 5"],
   ["Spring: 11 - 12", "Draft evaluation and conclusions; address comments; submit.", "Final dissertation and code"],
  ]),
("bt", "If a driver or package update breaks the Windows stack, the pinned versions are reinstalled, with WSL2 as a fallback. If a larger adapter trains much slower than measured, fewer rounds fit, but every kept round still adds a point to the curve. If an improvement sits inside run-to-run noise, more seeds are run before concluding. Every change of plan is dated in the decision log."),
]

ACC = " (accessed Oct. 17, 2026)."
REFS = {
 "lora": ("E. J. Hu *et al.*, \"LoRA: Low-rank adaptation of large language models,\" in *Proc. Int. Conf. Learning Representations (ICLR)*, 2022.", "https://arxiv.org/abs/2106.09685"),
 "qlora": ("T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, \"QLoRA: Efficient finetuning of quantized LLMs,\" in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023, pp. 10088-10115.", "https://arxiv.org/abs/2305.14314"),
 "biderman": ("D. Biderman *et al.*, \"LoRA learns less and forgets less,\" *Trans. Machine Learning Research*, Aug. 2024.", "https://arxiv.org/abs/2405.09673"),
 "ibrahim": ("A. Ibrahim *et al.*, \"Simple and scalable strategies to continually pre-train large language models,\" *Trans. Machine Learning Research*, 2024.", "https://arxiv.org/abs/2403.08763"),
 "yang": ("S. Yang, W.-L. Chiang, L. Zheng, J. E. Gonzalez, and I. Stoica, \"Rethinking benchmark and contamination for language models with rephrased samples,\" arXiv preprint arXiv:2311.04850, 2023.", "https://arxiv.org/abs/2311.04850"),
 "longlora": ("Y. Chen *et al.*, \"LongLoRA: Efficient fine-tuning of long-context large language models,\" in *Proc. Int. Conf. Learning Representations (ICLR)*, 2024.", "https://arxiv.org/abs/2309.12307"),
 "ruler": ("C.-P. Hsieh *et al.*, \"RULER: What's the real context size of your long-context language models?\" in *Proc. Conf. Language Modeling (COLM)*, 2024.", "https://arxiv.org/abs/2404.06654"),
 "swebench": ("C. E. Jimenez *et al.*, \"SWE-bench: Can language models resolve real-world GitHub issues?\" in *Proc. Int. Conf. Learning Representations (ICLR)*, 2024.", "https://arxiv.org/abs/2310.06770"),
 "qwen": ("Qwen Team, \"Qwen3.8-27B,\" Hugging Face model card, Aug. 2026.", "https://huggingface.co/Qwen/Qwen3.8-27B", ACC),
 "verified": ("N. Chowdhury *et al.*, \"Introducing SWE-bench Verified,\" OpenAI, Aug. 13, 2024.", "https://openai.com/index/introducing-swe-bench-verified/", ACC),
 "rocm": ("AMD, \"Compatibility matrix,\" ROCm Documentation, 2026.", "https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html", ACC),
 "peft": ("S. Mangrulkar, S. Gugger, L. Debut, Y. Belkada, S. Paul, and B. Bossan, \"PEFT: State-of-the-art parameter-efficient fine-tuning methods,\" GitHub repository, 2022.", "https://github.com/huggingface/peft", ACC),
 "bnb": ("bitsandbytes Foundation, \"Installation guide, v0.50.2,\" Hugging Face documentation, Aug. 2026.", "https://huggingface.co/docs/bitsandbytes/v0.50.2/installation", ACC),
 "llamacpp": ("G. Gerganov *et al.*, \"llama.cpp: LLM inference in C/C++,\" GitHub repository.", "https://github.com/ggml-org/llama.cpp", ACC),
 "lcb": ("N. Jain *et al.*, \"LiveCodeBench: Holistic and contamination free evaluation of large language models for code,\" in *Proc. Int. Conf. Learning Representations (ICLR)*, 2025, pp. 58791-58831.", "https://arxiv.org/abs/2403.07974"),
 "longcode": ("S. Rando *et al.*, \"LongCodeBench: Evaluating coding LLMs at 1M context windows,\" arXiv preprint arXiv:2505.07897, 2025.", "https://arxiv.org/abs/2505.07897"),
 "repoqa": ("J. Liu *et al.*, \"RepoQA: Evaluating long context code understanding,\" in *ICML Workshop on Long-Context Foundation Models*, 2024.", "https://arxiv.org/abs/2406.06025"),
 "unsloth": ("D. Han, M. Han, and Unsloth team, \"Unsloth,\" GitHub repository, 2023.", "https://github.com/unslothai/unsloth", ACC),
 "cce": ("E. Wijmans, B. Huval, A. Hertzberg, V. Koltun, and P. Kr\u00e4henb\u00fchl, \"Cut your losses in large-vocabulary language models,\" in *Proc. Int. Conf. Learning Representations (ICLR)*, 2025.", "https://arxiv.org/abs/2411.09009"),
 "unslothq38": ("Unsloth, \"Qwen3.8: How to fine-tune,\" Unsloth documentation, 2026.", "https://unsloth.ai/docs/models/qwen3.8/train", ACC),
 "unslothq35": ("Unsloth, \"Qwen3.5 fine-tuning guide,\" Unsloth documentation, 2026.", "https://unsloth.ai/docs/models/qwen3.5/fine-tune", ACC),
 "alphacode": ("Y. Li *et al.*, \"Competition-level code generation with AlphaCode,\" *Science*, vol. 378, no. 6624, pp. 1092-1097, 2022.", "https://arxiv.org/abs/2203.07814"),
 "restem": ("A. Singh *et al.*, \"Beyond human data: Scaling self-training for problem-solving with language models,\" arXiv preprint arXiv:2312.06585, 2023.", "https://arxiv.org/abs/2312.06585"),
 "wsd": ("A. H\u00e4gele, E. Bakouch, A. Kosson, L. Ben Allal, L. Von Werra, and M. Jaggi, \"Scaling laws and compute-optimal training beyond fixed training durations,\" arXiv preprint arXiv:2405.18392, 2024.", "https://arxiv.org/abs/2405.18392"),
 "yuan": ("Z. Yuan *et al.*, \"Scaling relationship on learning mathematical reasoning with large language models,\" arXiv preprint arXiv:2308.01825, 2023.", "https://arxiv.org/abs/2308.01825"),
 "evalplus": ("J. Liu, C. S. Xia, Y. Wang, and L. Zhang, \"Is your code generated by ChatGPT really correct? Rigorous evaluation of large language models for code generation,\" in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023.", "https://arxiv.org/abs/2305.01210"),
 "oci": ("W. U. Ahmad *et al.*, \"OpenCodeInstruct: A large-scale instruction tuning dataset for code LLMs,\" arXiv preprint arXiv:2504.04030, 2025.", "https://arxiv.org/abs/2504.04030"),
 "deepseek": ("DeepSeek-AI, \"DeepSeek-R1-0528,\" Hugging Face model card, 2025.", "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528", ACC),
 "ocr": ("W. U. Ahmad *et al.*, \"OpenCodeReasoning: Advancing data distillation for competitive coding,\" arXiv preprint arXiv:2504.01943, 2025.", "https://arxiv.org/abs/2504.01943"),
 "dsv4": ("DeepSeek-AI, \"DeepSeek-V4-Pro,\" Hugging Face model card, 2026.", "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro", ACC),
 "tachibana": ("sequelbox, \"Tachibana4-DeepSeek-V4-Pro,\" Hugging Face dataset card, 2026.", "https://huggingface.co/datasets/sequelbox/Tachibana4-DeepSeek-V4-Pro", ACC),
 "banaxi": ("Banaxi-Tech, \"Deepseek-V4-Reasoning-Code-2500,\" Hugging Face dataset card, 2026.", "https://huggingface.co/datasets/Banaxi-Tech/Deepseek-V4-Reasoning-Code-2500", ACC),
 "nemomath": ("NVIDIA, \"Nemotron-SFT-Math-v4,\" Hugging Face dataset card, 2026.", "https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v4", ACC),
 "agentset": ("Public agent-session dataset card, Hugging Face, 2026.", "https://huggingface.co/datasets", ACC),
 "mmlupro": ("Y. Wang *et al.*, \"MMLU-Pro: A more robust and challenging multi-task language understanding benchmark,\" in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 37, 2024.", "https://arxiv.org/abs/2406.01574"),
 "swegym": ("J. Pan *et al.*, \"Training software engineering agents and verifiers with SWE-Gym,\" in *Proc. 42nd Int. Conf. Machine Learning (ICML)*, PMLR vol. 267, 2025, pp. 47717-47737.", "https://proceedings.mlr.press/v267/pan25g.html"),
}
