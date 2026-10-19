# Anneal
## Fine-Tuning an Open-Weight Coding Model on Hardware I Already Own

### What's the problem, and what will you deliver?

**Problem.** From the reading I've done so far, open-weight language models in the 70B–120B range, things like gpt-oss-120b or the larger Qwen3 models, seem to be roughly the biggest models you can realistically run, and now maybe even train, on hardware a normal person could own. That's mostly down to newer unified-memory chips like AMD's Ryzen AI Max+ 395 ("Strix Halo"), which can give the GPU access to up to 128GB of memory instead of the 8–24GB you'd get on a normal graphics card. From what I've read and tried myself, these open models are noticeably worse at coding than the big closed ones (GPT-5-class models), especially anything involving tool use or working across a whole codebase rather than a single function. Fine-tuning with LoRA/QLoRA is supposed to be the standard way to specialise a model like this without retraining the whole thing, and proper support for training on this kind of AMD chip (ROCm) is very new, it only really started working reliably in 2026. As far as I can tell, there isn't much written down yet about how much fine-tuning can actually close that coding gap when you're limited to consumer hardware, or what it realistically costs in time and effort to get there. That's what I want to find out, partly because it's a genuinely open question and partly because I don't yet know the answer myself.

**Aim.** To see how much targeted fine-tuning can actually improve a model's coding ability, using only hardware I already own, and to measure that properly with a before/after comparison rather than just going on impressions. I'm planning to start with a smaller model, somewhere in the 20B–40B range, so I know the training will realistically fit and complete. If that goes well and I have time and understanding left over, I want to try pushing up toward 70B, which by my current (rough, and probably not fully correct yet) understanding of the memory requirements is close to the limit of what would fit for training on 64GB. I expect my sense of exactly where that limit sits will get a lot more accurate once I've actually run some experiments.

**Artefact.** I'm aiming to end up with:
- A fine-tuned model (a LoRA adapter, plus a merged and quantised export) that's noticeably better at coding tasks than the base model it started from.
- A written-up, repeatable process for how I got there, dataset prep, training setup, exporting the result, so someone else with similar hardware could follow it.
- A test setup that runs the base model and the fine-tuned model through the same coding questions and reports the difference.
- An honest account of what worked, what didn't, and where the hardware got in the way, since I expect a fair amount of the learning to be in the failures.

**What I'm planning to cover:**
- One base model, mainly in the 20B–40B range (I'll pick between a dense or a mixture-of-experts model once I understand the trade-offs better), with a 70B attempt as a stretch goal if time and memory allow. I might also try running a 120B-class model just for inference, to see how big the remaining gap is, but I don't expect to be able to fine-tune something that size, its full set of weights alone is close to or bigger than what I have to work with.
- Fine-tuning with LoRA/QLoRA rather than a full retrain, since a full retrain of a model this size isn't realistic on this hardware.
- Testing on established coding benchmarks (things like HumanEval and MBPP), plus a small set of my own test questions that feel more like real coding tasks.
- Enough variation between experiments (adapter size, which parts of the model I train, what data I use) that I can actually say something about what mattered and what didn't.

**What I'm deliberately not doing:**
- Training a model from nothing, or doing a full fine-tune of every parameter, neither is realistic in the time or on this hardware.
- Anything requiring multiple GPUs or datacenter-scale setups.
- Fine-tuning for anything other than coding, general chat, reasoning, safety work, etc. are all out of scope.
- Building a brand new benchmark from scratch, I'll be using and adapting existing ones rather than inventing my own measurement system.
- Comparing lots of different base models. I'd rather do one properly than several shallowly.

**Related work.** From some initial reading, this connects to two areas I'll need to properly get to grips with rather than just skim: parameter-efficient fine-tuning methods like LoRA and QLoRA, and existing code-specific fine-tuning work such as Code Llama and StarCoder. Almost everything I've found so far assumes people are using proper datacenter GPUs, I haven't been able to find much that specifically covers doing this on unified-memory consumer hardware, which is part of why I think it's worth doing. I don't expect to come up with a new fine-tuning method here. The contribution, if there is one, is more likely to be an honest, tested account of what existing methods actually achieve on hardware like this, since that doesn't seem to be written down anywhere properly yet.

### Why is this an interesting problem?

There's real demand for a fully local coding assistant, from individual developers to organisations that don't want to send code to a third-party API, and the tools for it have moved fast. Whether fine-tuning actually closes the gap to a hosted model still feels like forum opinion rather than tested fact, which is what makes it interesting to me.

I'm drawn to it because it mixes two things I don't have much experience with: the systems side (memory limits, quantisation, a training stack that's still rough) and the ML side (choosing good data, measuring improvement properly). Every run gives me something to try myself, which should keep a year-long project feeling real. I think it'd help other students with similar hardware, small teams weighing up a local assistant, and mainly me, as a proper way into applied ML.

### What is the main challenge?

The biggest risk I can see is that training anything at the top end of what I'm attempting is right at the edge of what this hardware can actually do. From rules of thumb I've picked up so far (which I expect to refine once I actually start), a 70B model at 4-bit precision would use most of the 64GB I have available, which wouldn't leave much room for everything else training needs on top of that. I don't fully understand this yet, a decent chunk of the early project time will probably go into actually confirming what's realistic rather than trusting my current back-of-envelope numbers, and adjusting the target model size if I've got it wrong.

Second, ROCm support for this specific chip is new, and I don't have experience with it yet. I'm expecting driver issues, missing features, or results that look wrong and I won't immediately know whether that's a bug in the software stack or a mistake I've made, and I'll need to learn how to tell the difference.

Third, I understand in principle that a fine-tuning dataset which is too narrow can make a model worse at everything except the narrow thing it was trained on, but I haven't built a dataset like this before, so getting the balance right is something I expect to learn by trial and error, checking results as I go rather than getting it right first time.

Fourth, I think evaluating this properly is harder than it sounds. From what I've read, it's easy to accidentally let benchmark questions leak into training data, and easy to get a result that looks good only because of one lucky run. Designing a test that actually means something, and being honest about how much to trust a small number of runs given how long each one takes, is something I'll need to think about carefully rather than assume I've got right by default.

### Resource requirements

**Hardware.** A GMKtec EVO-X2 (AMD Ryzen AI Max+ 395 "Strix Halo", 64GB unified LPDDR5X memory), which I already own, will be the only compute I use for both training and testing, so no departmental hardware or compute access should be needed. Based on what I've read so far, the amount of that 64GB actually usable for training is around 48GB on Windows (it appears to cap the GPU's share at about 75% of total memory) or possibly closer to 56GB on Linux with ROCm, though I haven't confirmed these numbers myself yet and expect to firm them up once I start. This limit is a big part of why I've scoped the model sizes the way I have above. Storage is 1TB, which I believe should be enough, but a single training run for a model in my target range could involve a full-precision download of 60GB+, plus a similarly sized merged copy afterwards, plus a smaller final export, so I'll need to be disciplined about deleting intermediate files rather than keeping everything "just in case." If that turns out not to be enough, a cheap external drive is a fallback I can sort out myself rather than something I'd need from the department.

**Software.** ROCm (AMD's equivalent of CUDA) with a compatible PyTorch build; Unsloth, which I understand has recently added support for fine-tuning on this specific chip; llama.cpp for converting and quantising models afterwards, and for running them locally to test; and an existing coding-benchmark tool (I'm currently looking at the BigCode evaluation harness and EvalPlus) rather than building my own. Everything here is free and open source.

**Datasets.** A publicly available, permissively licensed coding dataset for the fine-tuning itself, I'll need to check the licence terms properly rather than assume, plus established public benchmarks (HumanEval, MBPP, and possibly a small slice of a more agentic coding benchmark) for testing. There's no personal or sensitive data involved anywhere in this project; the main thing I need to be careful about is making sure any training data I use is actually licensed for this kind of use.
