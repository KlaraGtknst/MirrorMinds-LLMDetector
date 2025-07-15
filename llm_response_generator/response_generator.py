import pandas as pd
import re
import time
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
import torch


class ResponseGenerator:
    def __init__(self, model_path: str, dtype=torch.bfloat16, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.detect_model_type(model_path):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, torch_dtype=dtype
            ).to(device)
            task = "text2text-generation"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.dtype
            ).to(device)
            task = "text-generation"
        self.text_generator = pipeline(
            task,
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,  # 0 if "cuda" in device else -1,
            torch_dtype=dtype,
        )

    def detect_model_type(self, model_path: str) -> bool:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path)
        return config.is_encoder_decoder

    @staticmethod
    def _clean_generated_text(text: str) -> str:
        pattern = r"(Generate a response between (\d+) to (\d+) words for the following question\.)((\ *)(\n*))*(.*?\?)(\?*)((\ *)(\n*))*"
        cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
        return cleaned_text.strip()

    def generate_responses(self, input_path: str, output_path: str) -> pd.DataFrame:
        df = pd.read_pickle(input_path)
        df["generated_text"] = ""
        df["generation_time"] = ""

        for index, row in df.iterrows():
            essay_word_count = int(1.1 * len(row["Essay"].split()))
            max_word = int(1.2 * essay_word_count)
            prompt = f"Generate a response between {essay_word_count} to {max_word} words for the following question. {row['prompt']}?"

            start_time = time.time()
            generated_texts = self.text_generator(
                prompt,
                max_length=max_word,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                num_beams=2,
                early_stopping=True,
            )
            end_time = time.time()

            generated_text = (
                generated_texts[0]["generated_text"] if generated_texts else None
            )
            cleaned_text = (
                self._clean_generated_text(generated_text) if generated_text else None
            )

            df.at[index, "generated_text"] = cleaned_text
            df.at[index, "generation_time"] = end_time - start_time

        df.to_pickle(output_path)
        print(f"Responses generated and saved to {output_path}")
        return df
