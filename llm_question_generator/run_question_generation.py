from llm_question_generator.question_generator import QuestionGenerator
from llm_question_generator.parser import apply_extraction
from llm_question_generator.question_config import QuestionGeneratorConfig
import pandas as pd
import torch
import time
from huggingface_hub import snapshot_download


def main():
    torch.cuda.empty_cache()
    start_time = time.time()
    model_path = snapshot_download(repo_id="google/flan-t5-small")

    config = QuestionGeneratorConfig(
        model_path=model_path,
        input_path="Datasets/sample_essays.pkl",
        output_path="Datasets/generated_questions.pkl",
    )

    df = pd.read_pickle(config.input_path)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    generator = QuestionGenerator(model_path=config.model_path, device=device)
    df = generator.generate_questions(df, config.essay_column, config.max_new_tokens)
    df = apply_extraction(df)
    df.to_pickle(config.output_path)

    print(f"Total script runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
