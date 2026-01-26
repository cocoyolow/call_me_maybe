from typing import Dict, List, Tuple, Any, Set
from llm_sdk import Small_LLM_Model
import json
import numpy as np
from numpy.typing import NDArray


def json_to_dict(path: str) -> Dict[str, int]:
    """convert the json file to a dict"""
    with open(path, 'r') as f:
        dictionnary = json.load(f)
    return dictionnary


def reverse_dict(vocab: Dict[str, int]) -> Dict[int, str]:
    """Reverse the keys and values of a dictionary."""
    return {v: k for k, v in vocab.items()}


def create_vocab_buckets(vocab: Dict[str, int]) -> \
        Dict[str, List[Tuple[int, str]]]:
    """
    Create a dictionary of buckets based on the first character of the tokens.
    """
    buckets: Dict[str, List[Tuple[int, str]]] = {}
    for t_str, t_id in vocab.items():
        if not t_str:
            continue
        first_char = t_str[0]
        if first_char not in buckets:
            buckets[first_char] = []
        buckets[first_char].append((t_id, t_str))
    return buckets


def get_masks(vocab: Dict[int, str]) -> Dict[str, Set[int]]:
    masks = {
        'digits': set(),
        'digits_minus': set(),
        'minus': set(),
        'digits_dot': set(),
        'dot': set(),
        'bool': set(),
    }

    for t_id, raw_t_str in vocab.items():
        t_str = raw_t_str.replace('Ġ', '').strip()
        if not t_str:
            continue

        if t_str in ["true", "false"]:
            masks['bool'].add(t_id)

        if all(c in "0123456789" for c in t_str):
            masks['digits'].add(t_id)

        elif t_str == ".":
            masks['dot'].add(t_id)
        elif t_str == "-":
            masks['minus'].add(t_id)

    masks['digits_minus'] = masks['digits'] | masks['minus']
    masks['digits_dot'] = masks['digits'] | masks['dot']
    return masks


def get_function_name(llm: Small_LLM_Model,
                      input_ids: List[int],
                      vocab_buckets: Dict[str, List[Tuple[int, str]]],
                      allowed_names: List[str],
                      vocab: Dict[int, str]) -> Tuple[str, List[int]]:
    """
    Construct the function name using AI model with constrained decoding.
    """
    current_name = ""

    while True:
        if current_name in allowed_names:
            break

        logits = llm.get_logits_from_input_ids(input_ids)

        np_full = np.full(len(logits), -float('inf'))

        potential_candidates = [n for n in allowed_names
                                if n.startswith(current_name)]
        if not potential_candidates:
            break

        valid_next_chars = set()
        for cand in potential_candidates:
            remainder = cand[len(current_name):]
            if remainder:
                valid_next_chars.add(remainder[0])

        for char in valid_next_chars:
            for token_id, token_str in vocab_buckets[char]:
                candidate_name = current_name + token_str
                for name in potential_candidates:
                    if name.startswith(candidate_name):
                        np_full[token_id] = logits[token_id]
                        break

        next_token_id = int(np.argmax(np_full))
        next_token_str = vocab[next_token_id]

        input_ids.append(next_token_id)
        current_name += next_token_str

    return (current_name, input_ids)


def create_system_prompt(definitions: List[Dict[str, Any]],
                         current_prompt: str) -> str:
    """creates the prompt for the model.\n
    This adds contex to the prompt, giving all the function definitions"""
    txt = "Available tools:\n"
    for tool in definitions:
        txt += f"- {tool}\n"
    txt += "\nAnswer the user query using the correct tool.\n"
    txt += "for each function, only fill the arguments needed."
    txt += '''finish str with '"' character, no '\n' character '''
    return txt + current_prompt


def ask_for_float(llm: Small_LLM_Model,
                  input_ids: List[int], masks: Dict[str, Set[int]],
                  vocab: Dict[int, str], stop_tokens: set[int]) -> List[int]:
    """asks the model for a float"""
    state = "START"

    digits_and_end: Set[int] = masks['digits'] | stop_tokens
    digits_dot_and_end: Set[int] = masks['digits_dot'] | stop_tokens
    allowed_indices = set()
    while True:
        if state == "START":
            allowed_indices = masks['digits_minus']

        elif state == "INT_PART":
            allowed_indices = digits_dot_and_end
        elif state == "AFTER_DOT":
            # digit only
            allowed_indices = masks['digits']
        elif state == "DECIMAL_PART":
            # digit or end char
            allowed_indices = digits_and_end

        logits = llm.get_logits_from_input_ids(input_ids)

        best_natural = max(allowed_indices, key=lambda i: logits[i])

        if state in ["DECIMAL_PART",
                     "INT_PART"] and best_natural in stop_tokens:
            break

        token_str = vocab[best_natural].replace('Ġ', '').strip()

        if token_str == "-":
            state = "START"
        elif token_str == ".":
            state = "AFTER_DOT"
        elif token_str.isdigit():
            if state == "START" or state == "INT_PART":
                state = "INT_PART"
            elif state == "AFTER_DOT" or state == "DECIMAL_PART":
                state = "DECIMAL_PART"

        input_ids.append(best_natural)

    return input_ids


def ask_for_int(llm, input_ids, masks, vocab, stop_tokens):
    """asks the model for an int"""
    state = "START"

    allowed_indices = set()
    has_digits = False
    while True:

        if state == "START":
            allowed_indices = masks['digits_minus']
        elif state == "BODY":
            allowed_indices = masks['digits'] | stop_tokens

        logits = llm.get_logits_from_input_ids(input_ids)

        next_token = max(allowed_indices, key=lambda i: logits[i])

        if not has_digits and next_token in masks['digits_minus']:
            has_digits = True

        if has_digits and next_token in stop_tokens:
            break

        input_ids.append(next_token)

        token_str = vocab[next_token].replace('Ġ', '').strip()
        if token_str == "-":
            state = "BODY"

    return input_ids


def ask_for_bool(llm, input_ids, masks, vocab):
    """asks the model for a boolean (true or false)"""
    logits = np.array(llm.get_logits_from_input_ids(input_ids))
    next_token = max(masks['bool'], key=lambda i: logits[i])
    input_ids.append(next_token)


def ask_for_str(llm, input_ids, vocab):
    """asks the model for a string"""
    quote_ids = llm.encode('"')
    input_ids.extend(quote_ids)

    while True:
        logits = np.array(llm.get_logits_from_input_ids(input_ids))

        next_token = int(np.argmax(logits))
        token_str = vocab[next_token]

        token_str = vocab[next_token].replace('Ġ', '').strip()
        input_ids.append(next_token)
        if len(token_str) and token_str[-1] == '"':
            break

    return input_ids


def start_generation(combined_data: Dict[str,
                     List[Dict[str, str]] |
                     Dict]) -> str:
    """Start the model generation"""
    llm = Small_LLM_Model()
    vocab_path: str = llm.get_path_to_vocabulary_json()
    vocab: Dict[str, int] = json_to_dict(vocab_path)
    reversed_vocab: Dict[int, str] = reverse_dict(vocab)
    vocab_buckets = create_vocab_buckets(vocab)
    COMMA = llm.encode(',')
    BRACE_CLOSE = llm.encode('}')
    masks_dict: Dict[str, set[int]] = get_masks(reversed_vocab)

    for i in range(len(combined_data['calls'])):
        cur_prompt: str = combined_data['calls'][i]['prompt']
        prompt: str = create_system_prompt(combined_data['defs'],
                                           cur_prompt)
        input_ids: List[int] = llm.encode(prompt)
        generated_index = len(input_ids)

        skeleton: str = '{\n\t"prompt": ' + \
            f'"{cur_prompt}"' + \
            ',\n' + '\t"fn_name": "'
        encoded_skeleton = llm.encode(skeleton)
        input_ids.extend(encoded_skeleton)
        allowed_names: List[str] = []
        for y in range(len(combined_data['defs'])):
            allowed_names.append(combined_data['defs'][y]['fn_name'])

        name_result: Tuple[str, List[int]] = get_function_name(
            llm, input_ids, vocab_buckets, allowed_names, reversed_vocab)
        function_name = name_result[0]
        input_ids = name_result[1]
        input_ids.extend(llm.encode('",\n\t"args": {'))
        function_index: int = allowed_names.index(function_name)
        function: Dict[str, Any] = combined_data['defs'][function_index]
        stop_tokens = {k for k, v in reversed_vocab.items() if v in [",", "}",
                                                                     "\n"]}

        for arg in function['args_names']:
            input_ids.extend(llm.encode(f'"{arg}": '))
            match function['args_types'][arg]:
                case 'float':
                    ask_for_float(
                        llm, input_ids, masks_dict, reversed_vocab,
                        stop_tokens)
                case 'int':
                    ask_for_int(
                        llm, input_ids, masks_dict, reversed_vocab,
                        stop_tokens)
                case 'str':
                    ask_for_str(llm, input_ids, reversed_vocab)
                case 'bool':
                    ask_for_bool(llm, input_ids, masks_dict, reversed_vocab)
            if arg != function['args_names'][-1]:
                input_ids.extend(COMMA)
        input_ids.extend(BRACE_CLOSE)
        input_ids.extend(llm.encode("\n}"))
        print(llm.decode(input_ids[generated_index:]))
    print("\nGeneration finished.")

    return llm.decode(input_ids[generated_index:])
