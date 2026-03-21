from typing import Dict, List, Any, Set
import json
import numpy as np
import sys
from llm_sdk import Small_LLM_Model


def json_to_dict(path: str) -> Any:
    """Convert the JSON file to a dictionary.

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


def get_masks(vocab: Dict[int, str]) -> Dict[str, Set[int]]:
    """Return a dictionary with the masks for each type of token.

    The masks are used to restrict the generation of the model,
    thus optimise performance and accuracy.

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
        t_str = raw_t_str.replace('Ġ', ' ')
        if not t_str:
            continue

        if not ('"' in t_str and t_str[-1] != '"'):
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
    """Generate an ultra-optimized function name.

    Pre-calculates valid token sequences and follows them.
    Fills the input_ids with the tokens of the result.

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
    """Create a prompt optimized for accuracy with strict copying rules.

    Args:
        definitions (List[Dict[str, Any]]): the function definitions

    Returns:
        str: the detailed system prompt
    """
    txt = "Available functions:\n"
    for f in definitions:
        txt += f"- {f['name']}\n"
    txt += "\nchoose one function:"
    return txt


def create_single_function_context(func_def: Dict[str, Any]) -> str:
    """Create the context for a single function.

    Args:
        func_def (Dict[str, Any]): the function definition

    Returns:
        str: the context for the function
    """
    txt = f"Function {func_def['name']}:\n"

    txt += f"Description: {func_def['description']}\n"
    txt += "Parameters:\n"
    for param_name, param_info in func_def.get('parameters', {}).items():
        desc = param_info.get('description', '')
        txt += f"- {param_name} ({param_info.get('type')}): {desc}\n"

    txt += "\nGenerate arguments for this function:"
    return txt


def ask_for_float(llm: Small_LLM_Model,
                  input_ids: List[int], masks: Dict[str, Set[int]],
                  vocab: Dict[int, str], stop_tokens: set[int]) -> None:
    """Ask the model for a float and restrict it to a limited set of tokens.

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
    nb_tokens: int = 0
    while True:
        if nb_tokens > 200:
            break
        if state == "START":
            allowed_indices = masks['digits_minus']
        elif state == "AFTER_MINUS":
            allowed_indices = masks['digits']
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
            if state == "START" \
                    or state == "INT_PART" \
                    or state == "AFTER_MINUS":
                state = "INT_PART"
            elif state == "AFTER_DOT" or state == "DECIMAL_PART":
                state = "DECIMAL_PART"

        nb_tokens += 1
        input_ids.append(best_natural)
        print(token_str, end='')
    if '.' not in result:
        input_ids.extend(llm.encode('.0')[0].tolist())
        print(".0", end='')


def ask_for_int(llm: Small_LLM_Model,
                input_ids: List[int], masks: Dict[str, Set[int]],
                stop_tokens: set[int]) -> None:
    """Ask the model for an int and restrict it to a limited set of tokens.

    Args:
        llm (Small_LLM_Model): the model to use for the generation
        input_ids (List[int]): the input ids to use for the generation
        masks (Dict[str, Set[int]]): the masks to use for the generation
        stop_tokens (set[int]): the stop tokens to use for the generation

    Returns:
        None
    """
    nb_tokens: int = 0
    state = "START"
    allowed_indices: Set[int] = masks['digits_minus']
    has_digits = False
    while True:
        if nb_tokens > 200:
            break
        logits = llm.get_logits_from_input_ids(input_ids)

        next_token = max(allowed_indices, key=lambda i: logits[i])

        if not has_digits and next_token in masks['digits']:
            has_digits = True

        if has_digits and next_token in stop_tokens:
            break

        token_str = llm.decode([next_token])
        input_ids.append(next_token)
        print(token_str, end='')
        nb_tokens += 1
        if state != "BODY":
            allowed_indices = masks['digits'] | stop_tokens
            state = "BODY"


def ask_for_str(llm: Small_LLM_Model,
                input_ids: List[int],
                masks_dict: Dict[str, Set[int]],
                vocab: Dict[int, str]) -> None:
    """Ask the model for a string and restrict it to a limited set of tokens.

    Args:
        llm (Small_LLM_Model): the model to use for the generation
        input_ids (List[int]): the input ids to use for the generation
        masks_dict (Dict[str, Set[int]]): the masks to use for the generation
        vocab (Dict[int, str]): the vocabulary to use for the generation

    Returns:
        None
    """
    nb_tokens: int = 0
    quote_ids = llm.encode('"')[0].tolist()
    input_ids.extend(quote_ids)
    valid_indexes = np.array(list(masks_dict['valid_str_chars']), dtype=int)
    while True:
        if nb_tokens > 200:
            break
        logits = np.array(llm.get_logits_from_input_ids(input_ids))

        local_index = int(np.argmax(logits[valid_indexes]))

        next_token = valid_indexes[local_index]
        token_str = vocab[next_token]

        token_str = vocab[next_token].replace('Ġ', ' ')
        if token_str.endswith('"'):
            break
        token_str_clean = token_str.replace('"', '')
        nb_tokens += 1
        print(token_str_clean, end='', flush=True)
        input_ids.append(next_token)
    input_ids.extend(quote_ids)


def generate_prompts(test_count: int) -> List[Dict[str, str]]:
    """Generate the prompts to use for the generation using real words.

    Args:
        test_count (int): the number of tests to generate

    Returns:
        List[Dict[str, str]]: the prompts to use for the generation
    """
    from faker import Faker
    import random
    prompts = []
    fake = Faker('en_US')

    for _ in range(test_count):
        prompts.append({
            "prompt": fake.sentence(nb_words=random.randint(4, 25))
        })
    return prompts


def start_generation(combined_data: Dict[str,
                     List[Dict[str, str]]],
                     llm_name: str | None = None,
                     test_count: int = 0) -> List[Dict[str, Any]]:
    """Start the model generation and return the result.

    Returns a list of all prompts results.

    Args:
        combined_data (Dict[str, List[Dict[str, str]]]): the data provided in
            the json file
        llm_name (str | None): the name of the llm to use
        test_count (int): the number of tests to generate to verify that
            the generation is correct

    Returns:
        List[Dict[str, Any]]: the result of the generation
    """

    final_result: List[Dict[str, Any]] = []
    if llm_name is None:
        llm = Small_LLM_Model()
    else:
        llm = Small_LLM_Model(llm_name)
    vocab_path: str = llm.get_path_to_vocab_file()
    vocab: Dict[str, int] = json_to_dict(vocab_path)
    reversed_vocab: Dict[int, str] = reverse_dict(vocab)
    COMMA = llm.encode(',')[0].tolist()
    PADDING = "\t   - > "
    PROMPT = llm.encode('{\n\t"prompt": ')[0].tolist()
    FN_NAME = llm.encode(',\n\t"name": "')[0].tolist()
    PARAMETERS = llm.encode(',\n\t"parameters": {')[0].tolist()
    BRACE_CLOSE = llm.encode('}')[0].tolist()
    masks_dict: Dict[str, set[int]] = get_masks(reversed_vocab)
    allowed_names: List[str] = []
    stop_tokens = {k for k, v in reversed_vocab.items() if v in [",", "}",
                                                                 "\n"]}
    base_context: str = create_system_prompt(combined_data['defs'])
    base_input_ids: List[int] = llm.encode(base_context)[0].tolist()
    for y in range(len(combined_data['defs'])):
        allowed_names.append(combined_data['defs'][y]['name'])

    if test_count > 0:
        prompts = generate_prompts(test_count)
        combined_data['calls'] = prompts
    nb_prompts = len(combined_data['calls'])
    for i in range(nb_prompts):
        cur_prompt: str = combined_data['calls'][i]['prompt']
        input_ids: List[int] = base_input_ids.copy()
        context_index = len(input_ids)
        input_ids.extend(PROMPT)
        escaped_prompt = json.dumps(cur_prompt)
        print(f"prompt {i + 1}/{nb_prompts}: \033[94m{escaped_prompt}\033[0m")
        input_ids.extend(llm.encode(escaped_prompt)[0].tolist())
        input_ids.extend(FN_NAME)
        function_name: str = get_function_name(
            llm, input_ids, allowed_names)
        input_ids.extend(llm.encode('"')[0].tolist())
        input_ids = input_ids[context_index:]

        input_ids.extend(PARAMETERS)

        print(f"\033[92m{PADDING}detected function: {function_name}\033[0m")

        function_index: int = allowed_names.index(function_name)

        function: Dict[str, Any] = combined_data['defs'][function_index]

        fn_context_str: str = create_single_function_context(function)

        fn_context: list[int] = llm.encode(fn_context_str)[0].tolist()

        context_index = len(fn_context)

        input_ids = fn_context + input_ids

        parameters = function['parameters']
        arg_names = list(parameters.keys())
        for arg in arg_names:
            arg_type = parameters[arg]['type']
            input_ids.extend(llm.encode(f'"{arg}": ')[0].tolist())

            match arg_type:
                case 'number':
                    print(
                        f"\033[92m{PADDING}{arg} (number): \033[0m",
                        end='',
                        flush=True)
                    ask_for_float(
                        llm, input_ids, masks_dict, reversed_vocab,
                        stop_tokens)
                case 'integer':
                    print(
                        f"\033[92m{PADDING}{arg} (integer): \033[0m",
                        end='',
                        flush=True)
                    ask_for_int(
                        llm, input_ids, masks_dict, stop_tokens)
                case 'string':
                    print(
                        f"\033[92m{PADDING}{arg} (string): \033[0m",
                        end='',
                        flush=True)
                    ask_for_str(llm, input_ids, masks_dict, reversed_vocab)
                case _:
                    print(f"Error: unknown type {arg_type}")

            print()
            if arg != arg_names[-1]:
                input_ids.extend(COMMA)
        print()
        input_ids.extend(BRACE_CLOSE)
        input_ids.extend(llm.encode("\n}")[0].tolist())
        final_result.append(json.loads(
            llm.decode(input_ids[context_index:])))

    print("\nGeneration finished.")

    return final_result
