import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from data_loader import MCQSample


@dataclass
class EvalRow:
    model_name: str
    prompt_type: str
    language: str
    split: str
    include_lecture: bool
    total: int
    valid_predictions: int
    accuracy: float
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


def compute_accuracy(gold: List[int], pred: List[Optional[int]]) -> Dict[str, float]:
    total = len(gold)
    valid = sum(1 for p in pred if p is not None)
    correct = sum(1 for g, p in zip(gold, pred) if p is not None and g == p)
    accuracy = (correct / valid) if valid > 0 else 0.0
    valid_gold = [g for g, p in zip(gold, pred) if p is not None]
    valid_pred = [p for p in pred if p is not None]
    
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    confusion = None
    
    if valid_pred and len(set(valid_gold)) > 0:
        try:
            precision = precision_score(valid_gold, valid_pred, average='weighted', zero_division=0)
            recall = recall_score(valid_gold, valid_pred, average='weighted', zero_division=0)
            f1 = f1_score(valid_gold, valid_pred, average='weighted', zero_division=0)
            confusion = confusion_matrix(valid_gold, valid_pred)
        except Exception as e:
            print(f"Warning: Metrics calculation failed: {e}")
    
    return {
        "total": total,
        "valid_predictions": valid,
        "correct": correct,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion,
    }


def save_experiment_rows(rows: List[EvalRow], output_csv: str) -> None:
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def save_predictions(
    samples: List[MCQSample],
    preds: List[Optional[int]],
    raw_outputs: List[str],
    path: str,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "split",
                "question",
                "choices",
                "gold_answer_idx",
                "pred_answer_idx",
                "is_correct",
                "raw_model_output",
            ]
        )
        for s, p, raw in zip(samples, preds, raw_outputs):
            writer.writerow(
                [
                    s.sample_id,
                    s.split,
                    s.question,
                    " | ".join(s.choices),
                    s.answer_idx,
                    p if p is not None else "",
                    int(p == s.answer_idx) if p is not None else 0,
                    raw,
                ]
            )


def print_error_analysis(
    samples: List[MCQSample],
    preds: List[Optional[int]],
    raw_outputs: List[str],
    max_examples: int = 5,
) -> None:
    correct_examples = []
    incorrect_examples = []
    invalid_examples = []
    for s, p, raw in zip(samples, preds, raw_outputs):
        if p is None:
            invalid_examples.append((s, raw))
            continue
        if p == s.answer_idx:
            correct_examples.append((s, p, raw))
        else:
            incorrect_examples.append((s, p, raw))

    print("\n=== Phân tích Lỗi ===")
    print("\n[OK] Ví dụ đúng:")
    for s, p, _ in correct_examples[:max_examples]:
        print(f"- ID={s.sample_id} | đáp_án_đúng={s.answer_idx} dự_đoán={p} | chủ_đề={s.subject}")
        print(f"  Câu hỏi: {s.question[:200]}")

    print("\n[ERROR] Ví dụ sai:")
    for s, p, raw in incorrect_examples[:max_examples]:
        print(f"- ID={s.sample_id} | đáp_án_đúng={s.answer_idx} dự_đoán={p} | chủ_đề={s.subject}")
        print(f"  Câu hỏi: {s.question[:200]}")
        print(f"  Đầu ra: {raw[:200]}")
        print("  Lý do có thể: thiếu kiến thức, hướng dẫn yếu, hoặc nhầm lẫn với các lựa chọn khác.")

    print("\n[WARNING] Ví dụ phân tích không hợp lệ:")
    for s, raw in invalid_examples[:max_examples]:
        print(f"- ID={s.sample_id} | đáp_án_đúng={s.answer_idx} | chủ_đề={s.subject}")
        print(f"  Câu hỏi: {s.question[:200]}")
        print(f"  Đầu ra: {raw[:200]}")
        print("  Lý do có thể: mô hình không tuân theo định dạng đầu ra.")

