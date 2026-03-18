from typing import Dict, List, Tuple, Any, Set
from llm_sdk import Small_LLM_Model
import json
import numpy as np
import sys

def json_to_dict(path: str) -> Any:
    """convert the json file to a dict

    Args:
        path (str): path to the json file

    Returns:
        Any: the json file as a dict
    """
    try:
        with open(path, 'r') as f:
            dictionnary = json.load(f)
    except Exception as e:
        err_type = e.__class__.__name__
        print(f'Error appened:\nError type: {err_type}\nError message: {e}')
        sys.exit(1)
    return dictionnary


def reverse_dict(vocab: Dict[str, int]) -> Dict[int, str]:
    """Reverse the keys and values of a dictionary.

    Args:
        vocab (Dict[str, int]): the dictionary to reverse

    Returns:
        Dict[int, str]: the reversed dictionary
    """
    return {v: k for k, v in vocab.items()}


def create_vocab_buckets(vocab: Dict[str, int]) -> \
        Dict[str, List[Tuple[int, str]]]:
    """
    Create a dictionary of buckets based on the first character of the tokens.

    Args:
        vocab (Dict[str, int]): the vocabulary to create the buckets from

    Returns:
        Dict[str, List[Tuple[int, str]]]: the buckets created from the vocab
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
    """Returns a dictionary with the masks for each type of token
    the masks are used to restrict the generation of the model,
    thus optimise performance and accuracy

    Args:
        vocab (Dict[int, str]): the vocabulary to create the masks from

    Returns:
        Dict[str, Set[int]]: the masks created from the vocab
    """
    masks: Dict[str, Set[int]] = {
        'digits': set(),
        'digits_minus': set(),
        'minus': set(),
        'digits_dot': set(),
        'dot': set(),
        'valid_str_chars': set(),
    }

    for t_id, raw_t_str in vocab.items():
        t_str = raw_t_str.replace('Ġ', '').strip()
        if not t_str:
            continue

        if not ('"' in t_str and t_str[-1] != '"'):
            if "\\" not in t_str:
                masks['valid_str_chars'].add(t_id)

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
                      input_ids: List[int], allowed_names: List[str]) -> str:
    """
    Ultra-optimized function name generation.
    Pre-calculates valid token sequences and follows them.
    fills the input_ids with the tokens of the result
    Args:
        llm (Small_LLM_Model): the model to use for the generation
        input_ids (List[int]): the input ids to use for the generation
        allowed_names (List[str]): the allowed names to use for the generation

    Returns:
        str: the name of the function
    """
    candidate_sequences = [llm.encode(name)[0].tolist()
                           for name in allowed_names]

    active_indices = list(range(len(allowed_names)))
    pos = 0

    while True:
        valid_next_tokens = set()
        token_to_candidates: Dict[int, List[int]] = {}

        for idx in active_indices:
            seq = candidate_sequences[idx]
            if pos < len(seq):
                t = seq[pos]
                valid_next_tokens.add(t)

                if t not in token_to_candidates:
                    token_to_candidates[t] = []
                token_to_candidates[t].append(idx)

        if not valid_next_tokens:
            break

        logits = np.array(llm.get_logits_from_input_ids(input_ids))

        candidates_arr = np.array(list(valid_next_tokens), dtype=int)
        best_local_idx = np.argmax(logits[candidates_arr])
        best_token = candidates_arr[best_local_idx]

        input_ids.append(best_token)

        active_indices = token_to_candidates[best_token]
        pos += 1

        matched_name = None
        for idx in active_indices:
            if len(candidate_sequences[idx]) == pos:
                matched_name = allowed_names[idx]
                break

        if matched_name:
            return matched_name

    return ""


def create_system_prompt(
        definitions: List[Dict[str, Any]]) -> str:
    """
    Creates a prompt optimized for accuracy with strict copying rules.

    Args:
        definitions (List[Dict[str, Any]]): the function definitions
        current_prompt (str): the current prompt

    Returns:
        str: the detailed system prompt
    """
    txt = "choose one function:"
    functions = ",".join([f["fn_name"] for f in definitions])
    txt += f"\n{functions}"

    return txt


def ask_for_float(llm: Small_LLM_Model,
                  input_ids: List[int], masks: Dict[str, Set[int]],
                  vocab: Dict[int, str], stop_tokens: set[int]) -> None:
    """asks the model for a float and restricts it to limited set of tokens

    Args:
        llm (Small_LLM_Model): the model to use for the generation
        input_ids (List[int]): the input ids to use for the generation
        masks (Dict[str, Set[int]]): the masks to use for the generation
        vocab (Dict[int, str]): the vocabulary to use for the generation
        stop_tokens (set[int]): the stop tokens to use for the generation

    Returns:
        None
    """
    state = "START"
    result: str = ""
    digits_and_end: Set[int] = masks['digits'] | stop_tokens
    digits_dot_and_end: Set[int] = masks['digits_dot'] | stop_tokens
    allowed_indices = set()

    while True:
        if state == "START":
            allowed_indices = masks['digits_minus']
        elif state == "AFTER_MINUS":
            allowed_indices = masks['digits_dot']
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
        result += token_str
        if token_str == "-":
            state = "AFTER_MINUS"
        elif token_str == ".":
            state = "AFTER_DOT"
        elif token_str.isdigit():
            if state == "START" or state == "INT_PART":
                state = "INT_PART"
            elif state == "AFTER_DOT" or state == "DECIMAL_PART":
                state = "DECIMAL_PART"

        input_ids.append(best_natural)
        print(token_str, end='')
    if '.' not in result:
        input_ids.extend(llm.encode('.0')[0].tolist())
        print(".0")


def ask_for_int(llm: Small_LLM_Model,
                input_ids: List[int], masks: Dict[str, Set[int]],
                stop_tokens: set[int]) -> None:
    """asks the model for an int and restricts it to limited set of tokens

    Args:
        llm (Small_LLM_Model): the model to use for the generation
        input_ids (List[int]): the input ids to use for the generation
        masks (Dict[str, Set[int]]): the masks to use for the generation
        stop_tokens (set[int]): the stop tokens to use for the generation

    Returns:
        None
    """
    state = "START"
    allowed_indices: Set[int] = masks['digits_minus']
    has_digits = False
    while True:
        logits = llm.get_logits_from_input_ids(input_ids)

        next_token = max(allowed_indices, key=lambda i: logits[i])

        if not has_digits and next_token in masks['digits']:
            has_digits = True

        if has_digits and next_token in stop_tokens:
            break

        input_ids.append(next_token)

        if state != "BODY":
            allowed_indices = masks['digits'] | stop_tokens
            state = "BODY"


def ask_for_str(llm: Small_LLM_Model,
                input_ids: List[int],
                masks_dict: Dict[str, Set[int]],
                vocab: Dict[int, str]) -> None:
    """asks the model for a string and restricts it to limited set of tokens

    Args:
        llm (Small_LLM_Model): the model to use for the generation
        input_ids (List[int]): the input ids to use for the generation
        masks_dict (Dict[str, Set[int]]): the masks to use for the generation
        vocab (Dict[int, str]): the vocabulary to use for the generation

    Returns:
        None
    """
    quote_ids = llm.encode('"')[0].tolist()
    input_ids.extend(quote_ids)
    valid_indexes = np.array(list(masks_dict['valid_str_chars']), dtype=int)
    while True:
        logits = np.array(llm.get_logits_from_input_ids(input_ids))

        local_index = int(np.argmax(logits[valid_indexes]))

        next_token = valid_indexes[local_index]
        token_str = vocab[next_token]

        token_str = vocab[next_token].replace('Ġ', '').strip()
        if token_str.endswith('"'):
            break
        input_ids.append(next_token)
    input_ids.extend(quote_ids)


def start_generation(combined_data: Dict[str,
                     List[Dict[str, str]]]) -> List[Dict[str, Any]]:
    """Start the model generation and return the result,
    a list of all prompts results

    Args:
        combined_data (Dict[str, List[Dict[str, str]]]): the
        data provided in the json file

    Returns:
        List[Dict[str, Any]]: the result of the generation
    """
    final_result: List[Dict[str, Any]] = []
    llm = Small_LLM_Model()
    vocab_path: str = llm.get_path_to_vocab_file()
    vocab: Dict[str, int] = json_to_dict(vocab_path)
    reversed_vocab: Dict[int, str] = reverse_dict(vocab)
    COMMA = llm.encode(',')[0].tolist()
    PROMPT = llm.encode('{\n\t"prompt": ')[0].tolist()
    FN_NAME = llm.encode(',\n\t"fn_name": "')[0].tolist()
    ARGS_MES = llm.encode(',\n\t"args": {')[0].tolist()
    BRACE_CLOSE = llm.encode('}')[0].tolist()
    masks_dict: Dict[str, set[int]] = get_masks(reversed_vocab)
    allowed_names: List[str] = []
    for y in range(len(combined_data['defs'])):
        allowed_names.append(combined_data['defs'][y]['fn_name'])

    nb_prompts = len(combined_data['calls'])
    for i in range(nb_prompts):
        cur_prompt: str = combined_data['calls'][i]['prompt']
        prompt: str = create_system_prompt(combined_data['defs'])
        input_ids: List[int] = llm.encode(prompt)[0].tolist()
        generated_index = len(input_ids)
        input_ids.extend(PROMPT)
        escaped_prompt = json.dumps(cur_prompt)
        print(f"prompt {i + 1}/{nb_prompts}: {escaped_prompt}")
        input_ids.extend(llm.encode(escaped_prompt)[0].tolist())
        input_ids.extend(FN_NAME)
        name_result: str = get_function_name(
            llm, input_ids, allowed_names)
        function_name = name_result
        input_ids.extend(llm.encode('"')[0].tolist())
        input_ids.extend(ARGS_MES)
        print("\t   - > detected function:", function_name, "\n")
        function_index: int = allowed_names.index(function_name)
        function: Dict[str, Any] = combined_data['defs'][function_index]
        stop_tokens = {k for k, v in reversed_vocab.items() if v in [",", "}",
                                                                     "\n"]}
        for arg in function['args_names']:
            input_ids.extend(llm.encode(f'"{arg}": ')[0].tolist())
            match function['args_types'][arg]:
                case 'number':
                    print(f"{arg} (number): ")
                    ask_for_float(
                        llm, input_ids, masks_dict, reversed_vocab,
                        stop_tokens)
                case 'integer':
                    print(f"{arg} (integer): ")
                    ask_for_int(
                        llm, input_ids, masks_dict, stop_tokens)
                case 'string':
                    print(f"{arg} (string): ")
                    ask_for_str(llm, input_ids, masks_dict, reversed_vocab)
                case _:
                    print(f"Error: unknown type {function['args_types'][arg]}")

            if arg != function['args_names'][-1]:
                input_ids.extend(COMMA)
        input_ids.extend(BRACE_CLOSE)
        input_ids.extend(llm.encode("\n}")[0].tolist())
        # import time as t
        final_result.append(json.loads(
            llm.decode(input_ids[generated_index:])))
        # t1 = t.time()
        # print(llm.decode(input_ids[generated_index:]))
        # t2 = t.time()
        # print(f"Time for call: {t2 - t1}")
    print("\nGeneration finished.")

    return final_result
