from dataclasses import dataclass


@dataclass
class QuestionGeneratorConfig:
    model_path: str
    input_path: str
    output_path: str
    essay_column: str = "Essay"
    max_new_tokens: int = 8192
