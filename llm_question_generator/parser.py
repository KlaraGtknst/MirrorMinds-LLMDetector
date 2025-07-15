import re
import pandas as pd

pattern = r"(Generate a question that encapsulates the main theme of the above essay\/text\.)((\ *)(\n*))*(.*?\?)"


def extract_question(text: str) -> str:
    match = re.search(pattern, text)
    return match.group(5) if match else None


def apply_extraction(
    df: pd.DataFrame, column: str = "generated_question"
) -> pd.DataFrame:
    df["prompt"] = df[column]  # FIXME: Why? .apply(extract_question)
    return df.drop(columns=[column])
