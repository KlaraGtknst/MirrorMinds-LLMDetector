from llm_question_generator.question_generator import QuestionGenerator
from llm_question_generator.parser import apply_extraction
from llm_question_generator.question_config import QuestionGeneratorConfig
from llm_response_generator.response_generator import ResponseGenerator
from llm_response_generator.response_config import ResponseGeneratorConfig
import pandas as pd
import torch
import time
from huggingface_hub import snapshot_download


def main():
    torch.cuda.empty_cache()
    start_time = time.time()
    model_path = snapshot_download(repo_id="google/flan-t5-small")

    question_config = QuestionGeneratorConfig(
        model_path=model_path,
        input_path="Datasets/sample_essays.pkl",
        output_path="Datasets/generated_questions.pkl",
    )
    response_config = ResponseGeneratorConfig(
        model_path=model_path,
        input_path=question_config.output_path,
        output_path="Datasets/generated_responses.pkl",
    )

    df = pd.read_pickle(question_config.input_path)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    generator = QuestionGenerator(model_path=question_config.model_path, device=device)
    df = generator.generate_questions(
        df, question_config.essay_column, question_config.max_new_tokens
    )
    df = apply_extraction(df)
    df.to_pickle(question_config.output_path)

    response_generator = ResponseGenerator(
        model_path=response_config.model_path, device=device
    )
    df = response_generator.generate_responses(
        response_config.input_path, response_config.output_path
    )

    print(f"Total script runtime: {time.time() - start_time:.2f} seconds")
    print("Generated questions and responses saved successfully.")
    print(df)


if __name__ == "__main__":
    main()
