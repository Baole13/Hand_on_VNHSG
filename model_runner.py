import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    do_sample: bool = False


class HFModelRunner:
    def __init__(self, model_name: str):
        print(f"\n📦 Đang tải mô hình: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if torch.cuda.is_available():
            device_map = "cuda"
            print(f"✅ GPU được phát hiện: {torch.cuda.get_device_name(0)}")
        else:
            device_map = "cpu"
            try:
                import psutil
                available_mem = psutil.virtual_memory().available / (1024**3)
                print(f"⚠️  GPU không có sẵn, sử dụng CPU (chậm hơn)")
                print(f"   RAM có sẵn: {available_mem:.1f} GB")
                if available_mem < 8:
                    print(f"⚠️  CẢNH BÁO: Bộ nhớ thấp! Mô hình có thể gặp sự cố.")
                    print(f"   Đề xuất: Sử dụng các mô hình nhỏ hơn (TinyLlama, DistilGPT2)")
            except ImportError:
                print(f"⚠️  GPU không có sẵn, sử dụng CPU (chậm hơn)")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,  
            device_map=device_map,
            trust_remote_code=True,
        )

    def generate_batch(self, prompts: List[str], gen_cfg: GenerationConfig) -> List[str]:
        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_cfg.max_new_tokens,
                    do_sample=gen_cfg.do_sample,
                    temperature=gen_cfg.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            prompt_lengths = [len(self.tokenizer.decode(x, skip_special_tokens=True)) for x in inputs["input_ids"]]
            cleaned = []
            for full_text, p_len in zip(decoded, prompt_lengths):
                cleaned.append(full_text[p_len:].strip() if len(full_text) > p_len else full_text.strip())
            return cleaned
        except Exception as e:
            print(f"❌ LỖI trong tạo batch: {e}")
            return [""] * len(prompts)


def parse_answer_index(text: str, num_choices: int) -> Optional[int]:
    """
    Robust parser for final answer extraction.
    Supports:
      - "Answer: 2"
      - "2"
      - "The answer is B"
      - "(C)"
    """
    if not text:
        return None

    # 1) Explicit "answer: <digit>"
    m = re.search(r"answer\s*[:\-]?\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < num_choices:
            return idx

    # 2) Any standalone digit (first valid)
    for m in re.finditer(r"\b(\d+)\b", text):
        idx = int(m.group(1))
        if 0 <= idx < num_choices:
            return idx

    # 3) A/B/C/D letters
    m = re.search(r"\b([A-Z])\b", text.upper())
    if m:
        idx = ord(m.group(1)) - ord("A")
        if 0 <= idx < num_choices:
            return idx

    return None


def predict_indices(
    runner: HFModelRunner,
    prompts: List[str],
    num_choices: List[int],
    gen_cfg: GenerationConfig,
    batch_size: int = 4,
) -> Tuple[List[Optional[int]], List[str]]:
    preds: List[Optional[int]] = []
    raw_outputs: List[str] = []

    for i in range(0, len(prompts), batch_size):
        p_batch = prompts[i : i + batch_size]
        n_batch = num_choices[i : i + batch_size]
        out_batch = runner.generate_batch(p_batch, gen_cfg)
        raw_outputs.extend(out_batch)
        for txt, n in zip(out_batch, n_batch):
            preds.append(parse_answer_index(txt, n))
    return preds, raw_outputs

