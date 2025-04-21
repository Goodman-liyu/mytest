import json
import os
import re
from typing import Dict, Tuple, Optional, List
from datasets import Dataset, DatasetDict
from opencompass.openicl.icl_evaluator import AccEvaluator
from .base import BaseDataset

class MTruthfulQADataset(BaseDataset):

    @staticmethod
    def load(path):
        dataset = []
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                dataset.append({
                    'question': item['question'],
                    'option_a': item['option_a'],
                    'option_b': item['option_b'],
                    'option_c': item['option_c'],
                    'option_d': item['option_d'],
                    'answer': item['answer'],
                })
        dataset = Dataset.from_list(dataset)
        return dataset
