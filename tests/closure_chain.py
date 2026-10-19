"""Print the wrapper chain around qwen3_5.torch_chunk_gated_delta_rule (freevar names vs cell contents),
after model load and again after an SFTTrainer is built, to find what breaks torch.compile's guards."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unsloth  # noqa: F401,E402
from unsloth import FastLanguageModel  # noqa: E402

import torch  # noqa: E402


def chain(fn, depth=0, max_depth=5):
    pad = "  " * depth
    code = getattr(fn, "__code__", None)
    names = code.co_freevars if code else ()
    cells = getattr(fn, "__closure__", None) or ()
    print(f"{pad}{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', type(fn).__name__)} "
          f"id={id(fn):x} freevars={names} cells={len(cells)}")
    if len(names) != len(cells):
        print(f"{pad}  !! freevars and closure cells differ in count")
    for i, cell in enumerate(cells):
        try:
            v = cell.cell_contents
        except ValueError:
            v = "<empty>"
        label = names[i] if i < len(names) else "?"
        print(f"{pad}  [{i}] {label} = {type(v).__name__}: {str(v)[:70]!r}")
        if callable(v) and getattr(v, "__closure__", None) and depth < max_depth:
            chain(v, depth + 2, max_depth)


def show(tag):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    print(f"===== {tag}")
    chain(mq.torch_chunk_gated_delta_rule)


if __name__ == "__main__":
    model, tok = FastLanguageModel.from_pretrained("Qwen/Qwen3.5-0.8B", max_seq_length=1024, load_in_4bit=True,
                                                   dtype=torch.bfloat16)
    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                             use_gradient_checkpointing="unsloth")
    show("after load + peft")
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    tok = getattr(tok, "tokenizer", tok)
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=Dataset.from_dict({"text": ["hi"] * 4}),
                         args=SFTConfig(output_dir=os.path.join(os.environ["TEMP"], "closure_chain"), max_steps=1,
                                        report_to="none", dataset_num_proc=1, bf16=True))
    show("after SFTTrainer(...)")
