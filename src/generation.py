from typing import Dict, Union
from llm_sdk import Small_LLM_Model
from .parser import FunctionDefinitionsValidator, FunctionCallsValidator
import json


def json_to_dict(path: str) -> Dict[str, int]:
    """convert the json file to a dict"""
    with open(path, 'r') as f:
        dictionnary = json.load(f)
    return dictionnary


def reverse_dict(vocab: Dict[str, int]) -> Dict[int, str]:
    """Reverse the keys and values of a dictionary."""
    return {v: k for k, v in vocab.items()}


def start_generation(combined_data: Dict[str,
                     List[Dict[str,str]] |
                     Dict]) -> str:
    """Start the model generation"""
    llm = Small_LLM_Model()
    vocab_path: str = llm.get_path_to_vocabulary_json()
    reversed_vocab: Dict[str, int] = json_to_dict(vocab_path)
    vocab: Dict[int, str] = reverse_dict(reversed_vocab)
    llm.encode(combined_data[0]['prompt'])
    return ''
