import torch
from llm_response_generator import ResponseGenerator, ResponseGeneratorConfig
from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id="google/flan-t5-small")
device = "mps" if torch.backends.mps.is_available() else "cpu"

config = ResponseGeneratorConfig(
    model_path=model_path,
    input_path="Datasets/generated_questions.pkl",
    output_path="Datasets/generated_responses.pkl",
    device=device,
)

generator = ResponseGenerator(model_path=config.model_path, device=config.device)
generator.generate_responses(config.input_path, config.output_path)
