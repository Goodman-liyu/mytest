import json
import os
import re
from typing import List
from datasets import Dataset, DatasetDict
from opencompass.openicl.icl_evaluator import AccEvaluator
from .base import BaseDataset

# def levenshtein_similarity(str1, str2):
#     dist = levenshtein_distance(str1.lower(), str2.lower())
#     max_len = max(len(str1), len(str2))
#     return 1 - dist / max_len


def extract_result_sentences3(text):
    matches = re.findall(r"<result>(.*?)</result>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text


class CipherBankDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        path = kwargs.get("path", None)
        cipher = kwargs.get("cipher", None)
        dataset = DatasetDict()
        f_path = os.path.join(path, "{}.jsonl".format(cipher))
        f = open(f_path, "r", encoding="utf-8")
        lines = f.readlines()
        objs = []
        for line in lines:
            obj = json.loads(line)
            objs.append(obj)
        out_dict_list = []
        for obj in objs:
            question = obj[cipher]
            reference = obj["plaintext"]
            new_obj = dict(ciphertext=question, plaintext=reference)
            out_dict_list.append(new_obj)
        dataset = Dataset.from_list(out_dict_list)
        return dataset


class CipherBankEvaluator(AccEvaluator):

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions: List, references: List) -> dict:
        if len(predictions) != len(references):
            raise ValueError("The number of predictions does not match the number of references")

        correct = 0
        for i in range(len(predictions)):

            prediction = predictions[i]
            reference = references[i]
            correct += compare_strings(prediction, reference)
        accuracy = round(correct / len(predictions) * 100, 2)
        return {"accuracy": accuracy}


def compare_strings(str1: str, str2: str, compare_numbers: bool = True) -> bool:
    str1 = str1.strip()
    str2 = str2.strip()
    if len(str1) != len(str2):
        return False

    if compare_numbers:
        return str1.lower() == str2.lower()
    else:
        for i in range(len(str1)):
            if str1[i].isdigit() and str2[i].isdigit():
                continue
            if str1[i].lower() != str2[i].lower():
                return False
        return True
