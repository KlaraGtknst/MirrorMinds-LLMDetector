import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    AutoModelForSeq2SeqLM,
)
import torch
import time


class QuestionGenerator:
    def __init__(self, model_path: str, dtype=torch.bfloat16, device="cpu"):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.detect_model_type(model_path):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, torch_dtype=dtype
            ).to(device)
            task = "text2text-generation"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype
            ).to(device)
            task = "text-generation"
        self.generator = pipeline(
            task,
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,  # 0,
            torch_dtype=dtype,
        )

    def detect_model_type(self, model_path: str) -> bool:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path)
        return config.is_encoder_decoder

    def generate_questions(
        self, df: pd.DataFrame, essay_column: str = "Essay", max_length: int = 8192
    ) -> pd.DataFrame:
        df = df.copy()
        df["generated_question"] = ""
        df["prompt_generation_time"] = ""

        for index, row in df.iterrows():
            prompt = f"{row[essay_column]}. Generate a question that encapsulates the main theme of the above essay/text."
            print()
            print(prompt[-200:])
            start_time = time.time()
            result = self.generator(prompt, max_length=max_length)
            print(result)
            generated = result[0]["generated_text"] if result else None
            print(f"Generated question: {generated}")
            end_time = time.time()

            df.at[index, "generated_question"] = generated
            df.at[index, "prompt_generation_time"] = end_time - start_time

        return df
