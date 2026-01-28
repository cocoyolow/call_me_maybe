from typing import Dict, List, Tuple, Any, Set
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
        'valid_str_chars': set(),
    }

    for t_id, raw_t_str in vocab.items():
        t_str = raw_t_str.replace('Ġ', '').strip()
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
                      input_ids: List[int],
                      vocab_buckets: Dict[str, List[Tuple[int, str]]],
                      allowed_names: List[str],
                      vocab: Dict[int, str]) -> str:
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

    return current_name


def create_system_prompt(
        definitions: List[Dict[str, Any]], current_prompt: str) -> str:
    """
    Creates a prompt optimized for accuracy with strict copying rules.
    """
    # 1. Rôle
    txt = "You are an expert data extraction agent. Call the correct function with PRECISE arguments.\n"

    # 2. Outils
    txt += "Available tools:\n"
    for defi in definitions:
        txt += f"- {json.dumps(defi)}\n"

    # 3. RÈGLES D'OR (Pour contrer les hallucinations)
    txt += "\n### STRICT RULES:\n"
    txt += "1. **SOURCE STRING**: Copy the source string WORD-FOR-WORD from the user prompt. Do NOT add '$' signs. Do NOT change numbers.\n"
    txt += "2. **NEGATIVE NUMBERS**: Keep the minus sign (e.g., -5).\n"
    txt += "3. **VALID JSON**: Ensure all brackets and quotes are closed.\n"

    # 4. REGEX CHEAT SHEET (Menu imposé)
    txt += "\n### REGEX PATTERNS (Copy these exact patterns):\n"
    txt += "- For **DIGITS/NUMBERS**: Use \"\\\\d+\"\n"
    txt += "- For **VOWELS**: Use \"[aeiouAEIOU]\" (Must have brackets [])\n"
    txt += "- For **SPECIFIC WORDS**: Use the word itself (e.g. \"cat\")\n"
    txt += "- For **DATES**: Use \"\\\\d{4}-\\\\d{2}-\\\\d{2}\"\n"
    txt += "- For **ANYTHING**: Use \".+\"\n"

    # 5. EXAMPLES (C'est ici qu'on corrige tes erreurs)
    txt += "\n### EXAMPLES:\n"

    # Ex 1: Force la copie exacte (Pas de $) + Regex Digits
    txt += "User: 'Substitute the digits in \"Order 552 count 30\" with \"#\"'\n"
    txt += "Assistant: {\n"
    txt += '  "fn_name": "fn_substitute_string_with_regex",\n'
    txt += '  "args": {"source_string": "Order 552 count 30", "regex": "\\\\d+", "replacement": "#"}\n'
    txt += "}\n\n"

    # Ex 2: Force les crochets pour les voyelles
    txt += "User: 'Replace all vowels in \"Hello World\" with *'\n"
    txt += "Assistant: {\n"
    txt += '  "fn_name": "fn_substitute_string_with_regex",\n'
    txt += '  "args": {"source_string": "Hello World", "regex": "[aeiouAEIOU]", "replacement": "*"}\n'
    txt += "}\n\n"

    # Ex 3: Nombres Négatifs
    txt += "User: 'Add -5 and 10'\n"
    txt += "Assistant: {\n"
    txt += '  "fn_name": "fn_add_numbers",\n'
    txt += '  "args": {"a": -5, "b": 10}\n'
    txt += "}\n\n"

    # Ex 4: Remplacement simple
    txt += "User: 'Replace \"dog\" with \"cat\" in \"My dog barks\"'\n"
    txt += "Assistant: {\n"
    txt += '  "fn_name": "fn_substitute_string_with_regex",\n'
    txt += '  "args": {"source_string": "My dog barks", "regex": "dog", "replacement": "cat"}\n'
    txt += "}\n"

    # 6. Prompt final
    txt += "\nNow, answer the user query:\n"
    txt += f"User: {current_prompt}\n"
    txt += "Assistant: "

    return txt


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

    return input_ids


def ask_for_str(llm, input_ids, masks_dict, vocab):
    """asks the model for a string"""
    quote_ids = llm.encode('"')
    input_ids.extend(quote_ids)

    while True:
        logits = llm.get_logits_from_input_ids(input_ids)

        next_token = max(
            masks_dict['valid_str_chars'],
            key=lambda i: logits[i])
        token_str = vocab[next_token]

        token_str = vocab[next_token].replace('Ġ', '').strip()
        if token_str.endswith('"'):
            break
        input_ids.append(next_token)
    input_ids.extend(quote_ids)
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
    PROMPT = llm.encode('{\n\t"prompt": ')
    FN_NAME = llm.encode(',\n\t"fn_name": "')
    ARGS_MES = llm.encode(',\n\t"args": {')
    BRACE_CLOSE = llm.encode('}')
    masks_dict: Dict[str, set[int]] = get_masks(reversed_vocab)
    allowed_names: List[str] = []
    for y in range(len(combined_data['defs'])):
        allowed_names.append(combined_data['defs'][y]['fn_name'])

    for i in range(1):
        cur_prompt: str = combined_data['calls'][i]['prompt']
        prompt: str = create_system_prompt(combined_data['defs'],
                                           cur_prompt)
        input_ids: List[int] = llm.encode(prompt)
        generated_index = len(input_ids)
        input_ids.extend(PROMPT)
        input_ids.extend(llm.encode(f'"{cur_prompt}"'))
        input_ids.extend(FN_NAME)
        name_result: Tuple[str, List[int]] = get_function_name(
            llm, input_ids, vocab_buckets, allowed_names, reversed_vocab)
        function_name = name_result
        input_ids.extend(ARGS_MES)
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
                    ask_for_str(llm, input_ids, masks_dict, reversed_vocab)

            if arg != function['args_names'][-1]:
                input_ids.extend(COMMA)
        input_ids.extend(BRACE_CLOSE)
        input_ids.extend(llm.encode("\n}"))
        print(llm.decode(input_ids[generated_index:]))
    print("\nGeneration finished.")

    return llm.decode(input_ids[generated_index:])
