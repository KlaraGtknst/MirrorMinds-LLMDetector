from setuptools import setup, find_packages


setup(
    name="llm_question_generator",
    version="0.1.0",
    description="Library for generating questions from text using LLMs",
    author="Shubham Gupta, Klara Gutekunst",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "transformers",
        "accelerate",
        "torch",
        "torchvision",
        "torchaudio",
    ],
    entry_points={
        "console_scripts": ["generate-questions=llm_question_generator.main:main"]
    },
    python_requires=">=3.8",
)
