import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class MCQSample:
    sample_id: str
    split: str
    question: str
    choices: List[str]
    answer_idx: int
    lecture: str = ""
    explanation: str = ""
    subject: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def _safe_read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_choices(raw_choices) -> List[str]:
    if isinstance(raw_choices, list):
        return [str(x).strip() for x in raw_choices]
    if isinstance(raw_choices, dict):
        ordered = sorted(raw_choices.items(), key=lambda kv: str(kv[0]))
        return [str(v).strip() for _, v in ordered]
    return []


def _extract_answer_idx(record: Dict) -> Optional[int]:
    candidate_keys = ["answer", "answer_idx", "label", "correct", "target"]
    for key in candidate_keys:
        if key in record:
            value = record[key]
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                value = value.strip()
                if value.isdigit():
                    return int(value)
                if len(value) == 1 and value.upper().isalpha():
                    return ord(value.upper()) - ord("A")
    return None


def _record_to_sample(record: Dict, split: str, sample_id: str) -> Optional[MCQSample]:
    question = str(record.get("question", "")).strip()
    choices = _normalize_choices(record.get("choices", []))
    answer_idx = _extract_answer_idx(record)

    if not question or not choices or answer_idx is None:
        return None
    if answer_idx < 0 or answer_idx >= len(choices):
        return None

    return MCQSample(
        sample_id=sample_id,
        split=split,
        question=question,
        choices=choices,
        answer_idx=answer_idx,
        lecture=str(record.get("lecture", "")).strip(),
        explanation=str(record.get("explanation", "")).strip(),
        subject=str(record.get("subject", "")).strip(),
    )


def _extract_question_choices_from_text(question_text: str) -> Tuple[str, List[str]]:
    """
    Parse question strings that embed choices as:
      "...?\nA. ...\nB. ...\nC. ...\nD. ..."
    """
    if not question_text:
        return "", []
    text = question_text.strip().replace("\r\n", "\n")
    opt_pattern = re.compile(r"(?m)^\s*([A-D])\.\s*(.+?)\s*$")
    matches = list(opt_pattern.finditer(text))
    if len(matches) >= 2:
        first_opt_start = matches[0].start()
        question_only = text[:first_opt_start].strip()
        option_map: Dict[str, str] = {}
        for m in matches:
            option_map[m.group(1).upper()] = m.group(2).strip()
        ordered = [option_map[k] for k in sorted(option_map.keys())]
        return question_only, ordered
    return text, []


def _record_to_sample_vnhsg(record: Dict, split: str, sample_id: str) -> Optional[MCQSample]:
    """
    VNHSG-style sample fields (case-sensitive in raw data):
      - ID
      - Question (includes A/B/C/D choices inline)
      - Choice (correct letter, e.g., 'C')
      - Explanation
    """
    question_raw = str(record.get("Question", "")).strip()
    if not question_raw:
        return None
    question, parsed_choices = _extract_question_choices_from_text(question_raw)
    if not parsed_choices:
        return None

    choice_letter = str(record.get("Choice", "")).strip().upper()
    if not choice_letter or len(choice_letter) != 1 or choice_letter < "A" or choice_letter > "Z":
        return None
    answer_idx = ord(choice_letter) - ord("A")
    if answer_idx < 0 or answer_idx >= len(parsed_choices):
        return None

    return MCQSample(
        sample_id=str(record.get("ID", sample_id)),
        split=split,
        question=question,
        choices=parsed_choices,
        answer_idx=answer_idx,
        lecture="",
        explanation=str(record.get("Explanation", "")).strip(),
        subject=str(record.get("Subject", "")).strip(),
    )


def _load_canonical_scienceqa(dataset_dir: Path) -> Optional[Dict[str, List[MCQSample]]]:
    """
    Canonical ScienceQA format usually includes:
      - problems.json
      - pid_splits.json (with train/val/test arrays of IDs)
    """
    problems_path = dataset_dir / "problems.json"
    splits_path = dataset_dir / "pid_splits.json"
    if not problems_path.exists() or not splits_path.exists():
        return None

    problems = _safe_read_json(problems_path)
    pid_splits = _safe_read_json(splits_path)
    out: Dict[str, List[MCQSample]] = {"train": [], "validation": [], "test": []}

    split_map = {
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
    }
    for raw_split, pids in pid_splits.items():
        if raw_split not in split_map:
            continue
        split = split_map[raw_split]
        for pid in pids:
            pid_str = str(pid)
            rec = problems.get(pid_str, {})
            rec["answer"] = rec.get("answer")
            sample = _record_to_sample(rec, split=split, sample_id=pid_str)
            if sample is not None:
                out[split].append(sample)
    return out


def _guess_split_from_path(path: Path) -> str:
    p = str(path).lower()
    if "train" in p:
        return "train"
    if "validation" in p or "valid" in p or "val" in p or "eval" in p:
        return "validation"
    if "test" in p:
        return "test"
    return "unknown"


def _load_recursive_json(dataset_dir: Path) -> Dict[str, List[MCQSample]]:
    """
    Fallback loader for folder structures that contain many JSON files.
    Works with any record that has:
      - question
      - choices
      - answer/label
    """
    out: Dict[str, List[MCQSample]] = {
        "train": [],
        "validation": [],
        "test": [],
        "unknown": [],
    }
    json_files = sorted(dataset_dir.rglob("*.json"))
    for file_path in json_files:
        try:
            obj = _safe_read_json(file_path)
        except Exception:
            continue

        split = _guess_split_from_path(file_path)
        if isinstance(obj, list):
            for i, rec in enumerate(obj):
                if not isinstance(rec, dict):
                    continue
                sample = _record_to_sample(rec, split=split, sample_id=f"{file_path.stem}_{i}")
                if sample is None:
                    sample = _record_to_sample_vnhsg(rec, split=split, sample_id=f"{file_path.stem}_{i}")
                if sample:
                    out[split].append(sample)
        elif isinstance(obj, dict):
            sample = _record_to_sample(obj, split=split, sample_id=file_path.stem)
            if sample is None:
                sample = _record_to_sample_vnhsg(obj, split=split, sample_id=file_path.stem)
            if sample:
                out[split].append(sample)
    return out


def resolve_dataset_dir(dataset_dir: str) -> Path:
    """
    Resolve dataset path robustly.
    If requested path does not exist, try common fallbacks under workspace.
    """
    p = Path(dataset_dir)
    if p.exists():
        return p

    candidates = [
        Path("Dataset"),
        Path("dataset"),
        Path.cwd() / "Dataset",
        Path.cwd() / "dataset",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Dataset directory not found: {dataset_dir}. "
        "Tried fallbacks: Dataset, dataset (relative to current working directory)."
    )


def inspect_dataset_structure(dataset_dir: str) -> Dict:
    path = resolve_dataset_dir(dataset_dir)

    canonical = _load_canonical_scienceqa(path)
    data = canonical if canonical is not None else _load_recursive_json(path)
    stats = {
        split: len(samples)
        for split, samples in data.items()
        if split in {"train", "validation", "test", "unknown"}
    }
    stats["dataset_dir"] = str(path.resolve())
    stats["format"] = "canonical_scienceqa" if canonical is not None else "recursive_json"
    stats["example_fields"] = [
        "sample_id",
        "question",
        "choices",
        "answer_idx",
        "lecture",
        "explanation",
        "subject",
    ]
    return stats


def load_dataset_splits(dataset_dir: str) -> Dict[str, List[MCQSample]]:
    path = resolve_dataset_dir(dataset_dir)
    canonical = _load_canonical_scienceqa(path)
    return canonical if canonical is not None else _load_recursive_json(path)


def validate_sample(sample: MCQSample, sample_idx: int = 0) -> Tuple[bool, List[str]]:
    """
    Xác thực một MCQSample đơn lẻ để kiểm tra chất lượng dữ liệu.
    Trả về: (is_valid, error_messages)
    """
    errors = []
    
    if not sample.sample_id:
        errors.append(f"Mẫu {sample_idx}: Thiếu sample_id")
    if not sample.question or len(sample.question.strip()) < 5:
        errors.append(f"Mẫu {sample_idx} ({sample.sample_id}): Câu hỏi quá ngắn hoặc trống")
    if not sample.choices or len(sample.choices) < 2:
        errors.append(f"Mẫu {sample_idx} ({sample.sample_id}): Ít hơn 2 lựa chọn")
    
    if sample.answer_idx < 0 or sample.answer_idx >= len(sample.choices):
        errors.append(f"Mẫu {sample_idx} ({sample.sample_id}): Chỉ số câu trả lời không hợp lệ {sample.answer_idx} (có {len(sample.choices)} lựa chọn)")
    
    for choice_idx, choice in enumerate(sample.choices):
        if not choice or len(choice.strip()) == 0:
            errors.append(f"Mẫu {sample_idx} ({sample.sample_id}): Lựa chọn trống tại chỉ số {choice_idx}")
    
    return len(errors) == 0, errors


def validate_dataset(samples: List[MCQSample], verbose: bool = True) -> Tuple[List[MCQSample], Dict]:
    valid_samples = []
    all_errors = []
    seen_ids = set()
    duplicates = []
    
    for idx, sample in enumerate(samples):
        is_valid, errors = validate_sample(sample, idx)
        
        if sample.sample_id in seen_ids:
            duplicates.append(sample.sample_id)
            errors.append(f"Mẫu {idx} ({sample.sample_id}): ID trùng lặp")
            is_valid = False
        seen_ids.add(sample.sample_id)
        
        if errors:
            all_errors.extend(errors)
        
        if is_valid:
            valid_samples.append(sample)
    
    report = {
        "total_samples": len(samples),
        "valid_samples": len(valid_samples),
        "invalid_samples": len(samples) - len(valid_samples),
        "duplicate_ids": len(duplicates),
        "errors": all_errors if verbose else len(all_errors),
    }
    
    if verbose:
        print(f"✅ Báo cáo Xác thực Dataset:")
        print(f"  Tổng cộng: {report['total_samples']}")
        print(f"  Hợp lệ: {report['valid_samples']}")
        print(f"  Không hợp lệ: {report['invalid_samples']}")
        print(f"  ID trùng lặp: {report['duplicate_ids']}")
        if all_errors and len(all_errors) <= 10:
            for err in all_errors[:10]:
                print(f"    - {err}")
        elif all_errors:
            for err in all_errors[:5]:
                print(f"    - {err}")
            print(f"    ... và {len(all_errors) - 5} lỗi khác")
    
    return valid_samples, report

