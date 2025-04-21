import json
import os
import re
from typing import Dict, Tuple, Optional, List
from datasets import Dataset, DatasetDict
from opencompass.openicl.icl_evaluator import AccEvaluator
from .base import BaseDataset


class KKDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        path = kwargs.get("path", None)
        category = kwargs.get("category", None)
        dataset = DatasetDict()
        f_path = os.path.join(path, "people{}_num100.jsonl".format(category[0]))
        f = open(f_path, "r", encoding="utf-8")
        lines = f.readlines()
        objs = []
        for line in lines:
            obj = json.loads(line)
            objs.append(obj)
        out_dict_list = []
        for obj in objs:
            question = obj["quiz"]
            reference = obj["solution_text_format"]
            new_obj = dict(quiz=question, answer=reference)
            out_dict_list.append(new_obj)
        dataset = Dataset.from_list(out_dict_list)
        return dataset


class KKEvaluator(AccEvaluator):

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions: List, references: List) -> dict:
        if len(predictions) != len(references):
            raise ValueError("The number of predictions does not match the number of references")

        correct = 0

        for i in range(len(predictions)):
            prediction = predictions[i]
            reference = references[i]
            gt_status = parse_solution_text_format(reference)  # ground truth status
            expected_names = list(gt_status.keys())  # expected names

            pred_status = parse_model_answer(prediction, expected_names)
            if pred_status:
                print(f"\n[Content Validation]")
                print(f"  Expected: {gt_status}")
                print(f"  Predicted: {pred_status}")

                if pred_status == gt_status:
                    correct += 1
        accuracy = round(correct / len(predictions) * 100, 2)
        return {"accuracy": accuracy}


def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.

    Args:
        solution_text: Formatted solution text from dataset

    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    print("\n[Ground Truth Parsing]")

    for line in solution_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        match = re.search(r"\b([A-Za-z]+)\b.*?\b(knight|knave)\b", line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")

    return status_dict


def extract_solution(solution_str: str) -> str:
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        Tuple containing (extracted_answer, processed_string)
    """

    # Extract final answer using XML-style tags
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if not matches:
        print("[Error] No valid answer tags found")
        return ""

    final_answer = matches[-1].group(1).strip()
    return final_answer


def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.

    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification

    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    print("\n[Model Answer Parsing]")
    print(f"  Expected characters: {expected_names}")

    knight_count = answer_text.lower().count("knight")
    knave_count = answer_text.lower().count("knave")

    print(f"  Number of predicted roles: {knight_count + knave_count}")
    if knight_count + knave_count != len(expected_names):
        print(
            f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}"
        )
        return None

    for name in expected_names:
        pattern = re.compile(rf"\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b", re.IGNORECASE)
        match = pattern.search(answer_text)

        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Error] Missing identification for {name}")
            return None

    return status_dict
