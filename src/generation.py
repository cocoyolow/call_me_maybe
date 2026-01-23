from typing import Dict, List, Tuple
from llm_sdk import Small_LLM_Model
import json
import numpy as np


def json_to_dict(path: str) -> Dict[str, int]:
    """convert the json file to a dict"""
    with open(path, 'r') as f:
        dictionnary = json.load(f)
    return dictionnary


def reverse_dict(vocab: Dict[str, int]) -> Dict[int, str]:
    """Reverse the keys and values of a dictionary."""
    return {v: k for k, v in vocab.items()}


def get_function_name(llm: Small_LLM_Model,
                      input_ids: List[int], vocab: Dict[int, str],
                      allowed_names: List[str]) -> Tuple[str, List[int]]:
    """
    Construct the function name using AI model with constrained decoding.
    Génère le nom token par token en interdisant les chemins invalides.
    """
    current_name = ""

    while True:
        logits = llm.get_logits_from_input_ids(input_ids)

        np_array = np.array(logits)
        np_full = np.full(len(np_array), -float('inf'))

        for token_id, token_str in vocab.items():
            candidate_name = current_name + token_str

            if any(name.startswith(candidate_name) for name in allowed_names):
                np_full[token_id] = np_array[token_id]

        next_token_id = int(np.argmax(np_array))
        next_token_str = vocab[next_token_id]

        input_ids.append(next_token_id)
        current_name += next_token_str

        if current_name in allowed_names:
            break

    return (current_name, input_ids)


def start_generation(combined_data: Dict[str,
                     List[Dict[str, str]] |
                     Dict]) -> str:
    """Start the model generation"""
    llm = Small_LLM_Model()
    vocab_path: str = llm.get_path_to_vocabulary_json()
    reversed_vocab: Dict[str, int] = json_to_dict(vocab_path)
    vocab: Dict[int, str] = reverse_dict(reversed_vocab)
    generated_ids: List[int] = []
    input_ids: List[int] = llm.encode(combined_data['calls'][0]['prompt'])
    result = ""

    skeleton: str = '{\n\t"prompt": ' + \
        f'"{combined_data["calls"][0]["prompt"]}"' + ',\n' + '\t"fn_name": '

    encoded_skeleton = llm.encode(skeleton)
    input_ids.extend(encoded_skeleton)
    generated_ids.extend(encoded_skeleton)

    allowed_names: List[str] = []
    for i in range(len(combined_data['defs'])):
        allowed_names.append(combined_data['defs'][i]['fn_name'])
    name_result: Tuple[str, List[int]] = get_function_name(
        llm, input_ids, vocab, allowed_names)
    generated_structure += name_result[0]
    input_ids = name_result[1]
    print("\nGeneration terminée.")

    return llm.decode(generated_ids)
