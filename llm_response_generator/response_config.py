from dataclasses import dataclass


@dataclass
class ResponseGeneratorConfig:
    model_path: str
    input_path: str
    output_path: str
    device: str = "cpu"  # default device, e.g. "cuda:0" or "mps" or "cpu"
