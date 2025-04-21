import json
import os
import re
from typing import Dict, Tuple, Optional, List
from datasets import Dataset, DatasetDict
from opencompass.openicl.icl_evaluator import AccEvaluator
from .base import BaseDataset


class CountDownDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        path = kwargs.get("path", None)
        file_name = kwargs.get("file_name", None)
        file_path = os.path.join(path, file_name)

        dataset = DatasetDict()
        out_dict_list = []

        f = open(file_path, "r", encoding="utf-8")
        for line in f:
            line = json.loads(line)
            numbers = line["nums"]
            target = line["target"]
            new_obj = dict(
                numbers=numbers, target=target, target_nums={"numbers": numbers, "target": target}
            )
            out_dict_list.append(new_obj)

        dataset = Dataset.from_list(out_dict_list)
        return dataset


class CountDownEvaluator(AccEvaluator):

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions: List, references: List) -> dict:
        print(predictions, references)

        if len(predictions) != len(references):
            raise ValueError("The number of predictions does not match the number of references")

        correct = 0
        for equation, ref in zip(predictions, references):

            if equation is None:
                continue

            target = ref["target"]
            numbers = ref["numbers"]
            print(numbers)
            # print(equation)
            print(target)
            print(type(target), type(numbers))

            if validate_equation(equation, numbers) == False:
                continue
            if evaluate_equation(equation) == target:
                correct += 1
        accuracy = round(correct / len(predictions) * 100, 2)
        return {"accuracy": accuracy}


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        # return None
        pass
    solution_str = solution_str.split("\n")[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None
