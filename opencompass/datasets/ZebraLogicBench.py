import json
import os
import re
from typing import Dict, Tuple, Optional, List
from datasets import Dataset, DatasetDict
from opencompass.openicl.icl_evaluator import AccEvaluator
from .base import BaseDataset


class zebralogicbenchDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        path = kwargs.get("path", None)
        question_format_file = kwargs.get("question_format_file", None)
        reference_file = kwargs.get("reference_file", None)
        question_format_file = os.path.join(path, question_format_file)
        reference_file = os.path.join(path, reference_file)
        dataset = DatasetDict()
        out_dict_list = []

        f1 = open(question_format_file, "r", encoding="utf-8")
        f2 = open(reference_file, "r", encoding="utf-8")

        for question_format, reference in zip(f1, f2):
            question_format = json.loads(question_format)
            reference = json.loads(reference)
            puzzle = question_format["puzzle"]
            json_template = question_format["solution"]
            json_template = apply_lgp_grid_template(json_template)
            answer = reference["solution"]

            new_obj = dict(puzzle=puzzle, json_template=json_template, answer=answer)
            out_dict_list.append(new_obj)

        dataset = Dataset.from_list(out_dict_list)
        return dataset


class zebralogicbenchEvaluator(AccEvaluator):

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions: List, references: List) -> dict:

        if len(predictions) != len(references):
            raise ValueError("The number of predictions does not match the number of references")

        correct_cells = 0
        total_cells = 0
        solved_puzzles = 0
        for prediction, solution in zip(predictions, references):

            solution_table, this_total_cells = to_table(solution)
            total_cells += this_total_cells

            if prediction is None or "solution" not in prediction or prediction["solution"] is None:
                continue
            prediction_table = prediction["solution"]

            this_correct_cells = 0  # number in the solution_table
            for house in solution_table:
                for column in solution_table[house]:
                    # if prediction_table[house][column] not exist then pass
                    if house in prediction_table and column in prediction_table[house]:
                        truth_cell = solution_table[house][column].lower().strip()
                        if (
                            prediction_table[house][column] is None
                            or len(prediction_table[house][column]) == 0
                        ):
                            continue
                        if type(prediction_table[house][column]) == list:
                            predicted_cell = prediction_table[house][column][0].lower().strip()
                        elif type(prediction_table[house][column]) == str:
                            predicted_cell = prediction_table[house][column].lower().strip()
                        else:
                            raise ValueError(
                                f"Unknown type: {type(prediction_table[house][column])}"
                            )
                        if truth_cell.lower().strip() == predicted_cell.lower().strip():
                            this_correct_cells += 1

            correct_cells += this_correct_cells
            if this_correct_cells == this_total_cells:
                solved_puzzles += 1
        return {
            "correct_cells": correct_cells,
            "accuracy": correct_cells / total_cells * 100,
            "puzzle_acc": solved_puzzles / len(predictions) * 100,
        }


def to_table(solution):
    this_total_cells = 0
    num_houses = len(solution["rows"])
    columns = solution["header"]
    table = {}
    for i in range(num_houses):
        table[f"House {i+1}"] = {columns[j]: solution["rows"][i][j] for j in range(1, len(columns))}
        this_total_cells += len(columns) - 1
    return table, this_total_cells


def extract_last_complete_json(s):
    # Stack to keep track of opening and closing braces
    stack = []
    last_json_start = None
    last_json_str = None

    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif char == "}":
            if stack:
                start = stack.pop()
                if not stack:
                    # Complete JSON object found
                    last_json_str = s[last_json_start : i + 1]
                    last_json_start = None

    # Load the last JSON object
    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass

    return None


def apply_lgp_grid_template(origin_json_template):
    num_houses = len(origin_json_template["rows"])
    columns = origin_json_template["header"]
    assert columns[0] == "House"
    json_template = {"reasoning": "___", "solution": {}}
    for i in range(num_houses):
        json_template["solution"][f"House {i+1}"] = {
            columns[j]: "___" for j in range(1, len(columns))
        }
    json_str = json.dumps(json_template, indent=4)
    return json_str
