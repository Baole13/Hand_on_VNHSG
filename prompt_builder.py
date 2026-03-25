from dataclasses import dataclass
from typing import List

from data_loader import MCQSample


@dataclass
class PromptConfig:
    prompt_type: str  
    include_lecture: bool = True
    language: str = "vi"  # Thay đổi từ "en" sang "vi" (Tiếng Việt)


def _format_choices(choices: List[str]) -> str:
    lines = []
    for i, c in enumerate(choices):
        lines.append(f"{i}. {c}")
    return "\n".join(lines)


def build_prompt(sample: MCQSample, cfg: PromptConfig) -> str:
    
    base_instruction = (
        "Bạn là một chuyên gia giải các câu hỏi trắc nghiệm về khoa học.\n"
        "Luôn tuân theo định dạng sau:\n"
        "1. Phân tích câu hỏi\n"
        "2. Xem xét từng lựa chọn cẩn thận\n"
        "3. Chọn câu trả lời tốt nhất\n"
        "4. CHỈ xuất ra số chỉ mục câu trả lời cuối cùng (0, 1, 2, hoặc 3)\n\n"
    )
    
    few_shot_examples = (
        "VÍ DỤ:\n"
        "Ví dụ 1:\n"
        "Câu hỏi: Bộ phận nào của tế bào được gọi là nhà máy sản xuất năng lượng?\n"
        "Lựa chọn:\n"
        "0. Nhân tế bào\n"
        "1. Ty thể\n"
        "2. Ribosome\n"
        "3. Bộ máy Golgi\n"
        "Đáp án: 1\n\n"
        "Ví dụ 2:\n"
        "Câu hỏi: Nguyên tố nào có số hiệu nguyên tử là 6?\n"
        "Lựa chọn:\n"
        "0. Oxy\n"
        "1. Nitơ\n"
        "2. Carbon\n"
        "3. Silicon\n"
        "Đáp án: 2\n\n"
    )

    question_block = (
        f"Câu hỏi:\n{sample.question}\n\n"
        f"Lựa chọn:\n{_format_choices(sample.choices)}\n\n"
    )

    lecture_block = ""
    if cfg.include_lecture and sample.lecture.strip():
        lecture_block = f"Bối cảnh/Bài giảng:\n{sample.lecture}\n\n"

    if cfg.prompt_type == "direct":
        return (
            base_instruction
            + few_shot_examples
            + question_block
            + "Phân tích và trả lời CHỈ với chỉ số câu trả lời (0, 1, 2, hoặc 3):\n"
            + "Đáp án: "
        )

    if cfg.prompt_type == "cot":
        return (
            base_instruction
            + few_shot_examples
            + question_block
            + "Hãy suy nghĩ từng bước:\n"
            + "1. Trước tiên, xác định câu hỏi đang yêu cầu gì.\n"
            + "2. Xem xét từng lựa chọn cẩn thận.\n"
            + "3. Loại bỏ các lựa chọn không đúng.\n"
            + "4. Chọn câu trả lời tốt nhất.\n\n"
            + "Đáp án cuối cùng (chỉ số nguyên):\n"
            + "Đáp án: "
        )

    if cfg.prompt_type == "context":
        return (
            base_instruction
            + few_shot_examples
            + lecture_block
            + question_block
            + "Sử dụng bối cảnh/bài giảng được cung cấp để trả lời câu hỏi.\n"
            + "Đáp án cuối cùng (chỉ số 0, 1, 2, hoặc 3):\n"
            + "Đáp án: "
        )

    raise ValueError(f"Loại prompt không được hỗ trợ: {cfg.prompt_type}")


