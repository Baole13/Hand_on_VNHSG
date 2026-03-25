import argparse
import random
import sys
from pathlib import Path
from typing import List

# Fix encoding for Vietnamese characters on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from transformers import MarianMTModel, MarianTokenizer

from data_loader import MCQSample, inspect_dataset_structure, load_dataset_splits, validate_dataset
from evaluator import (
    EvalRow,
    compute_accuracy,
    print_error_analysis,
    save_experiment_rows,
    save_predictions,
)
from model_runner import GenerationConfig, HFModelRunner, predict_indices
from prompt_builder import PromptConfig, build_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="ScienceQA LLM evaluation pipeline")
    parser.add_argument("--dataset_dir", type=str, default="Dataset/ScienceQA")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--models", nargs="+", default=["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
    parser.add_argument("--prompt_types", nargs="+", default=["direct", "cot", "context"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/scienceqa")
    parser.add_argument("--include_lecture", action="store_true")
    parser.add_argument("--enable_vi_eval", action="store_true")
    parser.add_argument("--vi_samples", type=int, default=12)
    parser.add_argument("--cot_inspection", action="store_true")
    return parser.parse_args()


# CPU-Compatible Model Sizes
CPU_SAFE_MODELS = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "1.1B (✅ CPU-safe)",
    "gpt2": "124M (✅ Very fast on CPU)",
    "distilgpt2": "82M (✅ Ultra-fast on CPU)",
}

LARGE_MODELS_WARNING = {
    "Qwen/Qwen2-7B-Instruct": "7B (❌ Needs GPU, ~14GB VRAM)",
    "meta-llama/Llama-2-7b": "7B (❌ Needs GPU, ~14GB VRAM)",
    "gpt2-medium": "355M (⚠️  Medium, may be slow)",
}


class ENVITranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-vi"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate_texts(self, texts: List[str], batch_size: int = 16) -> List[str]:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            generated = self.model.generate(**inputs, max_new_tokens=256)
            out.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return out

    def translate_sample(self, sample: MCQSample) -> MCQSample:
        pieces = [sample.question] + sample.choices + [sample.lecture, sample.explanation]
        translated = self.translate_texts(pieces)
        q_vi = translated[0]
        choices_vi = translated[1 : 1 + len(sample.choices)]
        lecture_vi = translated[1 + len(sample.choices)]
        explanation_vi = translated[2 + len(sample.choices)]
        return MCQSample(
            sample_id=f"{sample.sample_id}_vi",
            split=sample.split,
            question=q_vi,
            choices=choices_vi,
            answer_idx=sample.answer_idx,
            lecture=lecture_vi,
            explanation=explanation_vi,
            subject=sample.subject,
        )


def build_prompts(samples: List[MCQSample], prompt_type: str, include_lecture: bool) -> List[str]:
    cfg = PromptConfig(prompt_type=prompt_type, include_lecture=include_lecture, language="vi")  # Thay đổi sang Tiếng Việt
    return [build_prompt(s, cfg) for s in samples]


def main():
    args = parse_args()
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = inspect_dataset_structure(args.dataset_dir)
    print("=== Kiểm tra Cấu trúc Dataset ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(
        "Thông báo: mỗi mẫu nên chứa question, choices, answer_idx, "
        "và các trường tùy chọn lecture/explanation."
    )

    split_data = load_dataset_splits(args.dataset_dir)
    split_samples = split_data.get(args.split, [])
    if not split_samples:
        raise ValueError(f"Không tải được mẫu nào cho split={args.split}. Kiểm tra định dạng dataset_dir.")

    print("\n=== Xác thực Dữ liệu ===")
    split_samples, val_report = validate_dataset(split_samples, verbose=True)
    if not split_samples:
        raise ValueError("Không có mẫu hợp lệ nào sau khi xác thực!")

    if args.max_samples > 0:
        split_samples = split_samples[: args.max_samples]

    rows: List[EvalRow] = []

    for model_name in args.models:
        # Check if model is too large for CPU
        if model_name in LARGE_MODELS_WARNING:
            print(f"\n❌ LỖI: {model_name} {LARGE_MODELS_WARNING[model_name]}")
            print(f"\nℹ️  Mô hình này yêu cầu GPU. Các mô hình an toàn cho CPU:")
            for safe_model, size in CPU_SAFE_MODELS.items():
                print(f"   • {safe_model:45} {size}")
            print(f"\nCách sử dụng: python main.py --models {list(CPU_SAFE_MODELS.keys())[0]}")
            continue  # Bỏ qua mô hình này
        
        print(f"\n📦 Đang tải mô hình: {model_name}")
        runner = HFModelRunner(model_name=model_name)
        gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens)

        for prompt_type in args.prompt_types:
            print(f"\n▶️  Đang chạy {model_name} | loại_prompt={prompt_type} | ngôn_ngữ=vi")
            prompts = build_prompts(
                split_samples,
                prompt_type=prompt_type,
                include_lecture=args.include_lecture or (prompt_type == "context"),
            )
            num_choices = [len(s.choices) for s in split_samples]
            gold = [s.answer_idx for s in split_samples]

            preds, raw_outputs = predict_indices(
                runner=runner,
                prompts=prompts,
                num_choices=num_choices,
                gen_cfg=gen_cfg,
                batch_size=args.batch_size,
            )
            metrics = compute_accuracy(gold, preds)
            
            print(f"\n📊 Độ chính xác: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['valid_predictions']} hợp lệ)")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1-Score: {metrics['f1']:.4f}")
            if metrics['confusion_matrix'] is not None:
                print(f"   Ma trận nhầm lẫn:")
                cm = metrics['confusion_matrix']
                for i, row in enumerate(cm):
                    print(f"     Lớp {i}: {row}")

            rows.append(
                EvalRow(
                    model_name=model_name,
                    prompt_type=prompt_type,
                    language="vi",  # Thay đổi sang Tiếng Việt
                    split=args.split,
                    include_lecture=args.include_lecture or (prompt_type == "context"),
                    total=int(metrics["total"]),
                    valid_predictions=int(metrics["valid_predictions"]),
                    accuracy=float(metrics["accuracy"]),
                    precision=float(metrics["precision"]),
                    recall=float(metrics["recall"]),
                    f1=float(metrics["f1"]),
                )
            )

            pred_csv = out_dir / f"predictions_{model_name.replace('/', '_')}_{prompt_type}_vi.csv"  # Thay đổi _en sang _vi
            save_predictions(split_samples, preds, raw_outputs, str(pred_csv))
            print_error_analysis(split_samples, preds, raw_outputs, max_examples=3)

            if args.cot_inspection and prompt_type == "cot":
                cot_path = out_dir / f"cot_outputs_{model_name.replace('/', '_')}.txt"
                with cot_path.open("w", encoding="utf-8") as f:
                    for s, raw in zip(split_samples, raw_outputs):
                        f.write(f"[ID] {s.sample_id}\n[Q] {s.question}\n[OUT] {raw}\n\n")
                print(f"Saved CoT outputs: {cot_path}")

        if args.enable_vi_eval:
            print("\nBuilding Vietnamese subset for EN vs VI evaluation...")
            translator = ENVITranslator()
            n = min(args.vi_samples, len(split_samples))
            vi_source = random.sample(split_samples, n)
            vi_samples = [translator.translate_sample(s) for s in vi_source]

            vi_dump = out_dir / "vi_samples_preview.csv"
            save_predictions(
                samples=vi_samples,
                preds=[None] * len(vi_samples),
                raw_outputs=[""] * len(vi_samples),
                path=str(vi_dump),
            )
            print(f"Saved 12-sample Vietnamese conversion preview: {vi_dump}")

            for prompt_type in args.prompt_types:
                print(f"\nRunning {model_name} | prompt={prompt_type} | language=vi")
                prompts_vi = build_prompts(
                    vi_samples,
                    prompt_type=prompt_type,
                    include_lecture=args.include_lecture or (prompt_type == "context"),
                )
                num_choices_vi = [len(s.choices) for s in vi_samples]
                gold_vi = [s.answer_idx for s in vi_samples]
                pred_vi, raw_vi = predict_indices(
                    runner=runner,
                    prompts=prompts_vi,
                    num_choices=num_choices_vi,
                    gen_cfg=gen_cfg,
                    batch_size=args.batch_size,
                )
                m_vi = compute_accuracy(gold_vi, pred_vi)
                print(f" VI Accuracy: {m_vi['accuracy']:.4f} | Precision: {m_vi['precision']:.4f} | Recall: {m_vi['recall']:.4f} | F1: {m_vi['f1']:.4f}")
                rows.append(
                    EvalRow(
                        model_name=model_name,
                        prompt_type=prompt_type,
                        language="vi",
                        split=args.split,
                        include_lecture=args.include_lecture or (prompt_type == "context"),
                        total=int(m_vi["total"]),
                        valid_predictions=int(m_vi["valid_predictions"]),
                        accuracy=float(m_vi["accuracy"]),
                        precision=float(m_vi["precision"]),
                        recall=float(m_vi["recall"]),
                        f1=float(m_vi["f1"]),
                    )
                )
                vi_pred_csv = out_dir / f"predictions_{model_name.replace('/', '_')}_{prompt_type}_vi.csv"
                save_predictions(vi_samples, pred_vi, raw_vi, str(vi_pred_csv))

    results_csv = out_dir / "results_summary.csv"
    if rows:
        save_experiment_rows(rows, str(results_csv))
        print(f"\nSaved results summary: {results_csv}")
    print("Done.")


if __name__ == "__main__":
    main()

