from typing import Dict, List, Tuple, Any, Optional
from llm_sdk import Small_LLM_Model
import json
import numpy as np

# Global cache to store the loaded model and related data
_MODEL_CACHE: Optional[Dict[str, Any]] = None


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
    buckets = {}
    for t_str, t_id in vocab.items():
        if not t_str:
            continue
        first_char = t_str[0]
        if first_char not in buckets:
            buckets[first_char] = []
        buckets[first_char].append((t_id, t_str))
    return buckets


def get_masks(vocab: Dict[int, str]) -> Dict[str, List[int]]:
    masks = {
        'digits': [],
        'minus': [],
        'dot': [],
        'bool': [],
    }

    for t_id, raw_t_str in vocab.items():
        t_str = raw_t_str.replace('Ġ', '').strip()
        if not t_str:
            continue

        if t_str in ["true", "false"]:
            masks['bool'].append(t_id)

        if all(c in "0123456789" for c in t_str):
            masks['digits'].append(t_id)

        elif t_str == ".":
            masks['dot'].append(t_id)
        elif t_str == "-":
            masks['minus'].append(t_id)
    return masks


def _get_cached_model_data() -> Dict[str, Any]:
    """
    Lazy loader for the model and vocabulary data.
    Ensures the model is loaded only once.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        llm = Small_LLM_Model()
        vocab_path: str = llm.get_path_to_vocabulary_json()
        vocab: Dict[str, int] = json_to_dict(vocab_path)
        reversed_vocab: Dict[int, str] = reverse_dict(vocab)
        vocab_buckets = create_vocab_buckets(vocab)
        masks = get_masks(reversed_vocab)
        
        _MODEL_CACHE = {
            'llm': llm,
            'vocab': vocab,
            'reversed_vocab': reversed_vocab,
            'vocab_buckets': vocab_buckets,
            'masks': masks
        }
    return _MODEL_CACHE


def get_function_name(llm: Small_LLM_Model,
                      input_ids: List[int],
                      vocab_buckets: Dict[str, List[Tuple[int, str]]],
                      allowed_names: List[str],
                      vocab: Dict[int, str]) -> Tuple[str, List[int]]:
    """
    Construct the function name using AI model with constrained decoding.
    Optimized to search only relevant tokens.
    """
    current_name = ""

    while True:
        if current_name in allowed_names:
            break

        logits = llm.get_logits_from_input_ids(input_ids)
        
        # Filter allowed names that start with our current progress
        potential_candidates = [n for n in allowed_names
                                if n.startswith(current_name)]
        if not potential_candidates:
            break

        # Collect valid next tokens directly
        valid_indices = []
        
        valid_next_chars = set()
        for cand in potential_candidates:
            remainder = cand[len(current_name):]
            if remainder:
                valid_next_chars.add(remainder[0])

        # Instead of iterating over everything, only look at buckets for valid next chars
        for char in valid_next_chars:
            if char in vocab_buckets:
                for token_id, token_str in vocab_buckets[char]:
                    candidate_token_full = current_name + token_str
                     # Check if this token actually advances towards a valid name
                    for name in potential_candidates:
                        if name.startswith(candidate_token_full):
                           valid_indices.append(token_id)
                           break
        
        if not valid_indices:
            break

        # Find the best token among ONLY the valid indices
        # Optimization: use python max over valid_indices, avoids full array scan
        best_token_id = max(valid_indices, key=lambda i: logits[i])
        
        next_token_str = vocab[best_token_id]
        
        input_ids.append(best_token_id)
        current_name += next_token_str

    return (current_name, input_ids)


def create_system_prompt(definitions: List[Dict[str, Any]],
                         current_prompt: str) -> str:
    """creates the prompt for the model"""
    txt = "Available tools:\n"
    for tool in definitions:
        txt += f"- {tool}\n"
    txt += "\nAnswer the user query using the correct tool.\n"
    return txt + current_prompt


def ask_for_float(llm, input_ids, masks, vocab, stop_tokens):
    """asks the model for a float"""
    state = "START"

    while True:
        allowed_indices = []

        if state == "START":
            allowed_indices = masks['digits'] + masks['minus']
        elif state == "INT_PART":
            allowed_indices = masks['digits'] + masks['dot']
        elif state == "AFTER_DOT":
            # digit only
            allowed_indices = masks['digits']
        elif state == "DECIMAL_PART":
            # digit or end char
            allowed_indices = masks['digits']

        logits = llm.get_logits_from_input_ids(input_ids)
        
        # Optimization: check if we should stop OR continue with allowed digits
        # Stop tokens are only valid in certain states
        candidates = allowed_indices
        if state in ["DECIMAL_PART", "INT_PART"]:
             candidates = allowed_indices + stop_tokens
        
        if not candidates:
            # Should not happen, but safe fallback
            break
            
        next_token = max(candidates, key=lambda i: logits[i])

        if state in ["DECIMAL_PART", "INT_PART"] and next_token in stop_tokens:
            break

        token_str = vocab[next_token].replace('Ġ', '').strip()

        if token_str == "-":
            state = "START"
        elif token_str == ".":
            state = "AFTER_DOT"
        elif token_str.isdigit():
            if state == "START" or state == "INT_PART":
                state = "INT_PART"
            elif state == "AFTER_DOT" or state == "DECIMAL_PART":
                state = "DECIMAL_PART"

        input_ids.append(next_token)

    return input_ids


def ask_for_int(llm, input_ids, masks, vocab, stop_tokens):
    """asks the model for an int"""
    state = "START"

    while True:
        allowed_indices = []

        if state == "START":
            allowed_indices = masks['digits'] + masks['minus']
        elif state == "BODY":
            allowed_indices = masks['digits']

        logits = llm.get_logits_from_input_ids(input_ids)

        candidates = allowed_indices
        if state == "BODY":
             candidates = allowed_indices + stop_tokens
        
        if not candidates:
            break

        next_token = max(candidates, key=lambda i: logits[i])

        if state == "BODY" and next_token in stop_tokens:
            break

        input_ids.append(next_token)

        token_str = vocab[next_token].replace('Ġ', '').strip()
        if token_str == "-":
            state = "START"
        else:
            state = "BODY"

    return input_ids


def ask_for_bool(llm, input_ids, masks, vocab):
    """asks the model for a boolean (true or false)"""
    logits = llm.get_logits_from_input_ids(input_ids)
    
    # Optimization: Only look at bool tokens
    allowed = masks['bool']
    if not allowed:
        return input_ids 
        
    next_token = max(allowed, key=lambda i: logits[i])
    input_ids.append(next_token)
    return input_ids


def ask_for_str(llm, input_ids, masks, vocab):
    """asks the model for a string"""
    quote_ids = llm.encode('"')
    input_ids.extend(quote_ids)

    while True:
        logits = np.array(llm.get_logits_from_input_ids(input_ids))
        # For strings, keeping full scan logic as masks are complex
        # But using numpy for speed on large vocab
        next_token = int(np.argmax(logits))
        token_str = vocab[next_token]

        if '"' in token_str:
            input_ids.append(next_token)
            break
        input_ids.append(next_token)

    return input_ids


def start_generation(combined_data: Dict[str,
                     List[Dict[str, str]] |
                     Dict]) -> str:
    """Start the model generation"""
    # Use cached model data
    model_data = _get_cached_model_data()
    llm = model_data['llm']
    vocab = model_data['vocab']
    reversed_vocab = model_data['reversed_vocab']
    vocab_buckets = model_data['vocab_buckets']
    masks_dict = model_data['masks']
    
    prompt: str = create_system_prompt(combined_data['defs'],
                                       combined_data['calls'][0]['prompt'])
    
    input_ids: List[int] = llm.encode(prompt)
    generated_index = len(input_ids)

    skeleton: str = '{\n\t"prompt": ' + \
        f'"{combined_data["calls"][0]["prompt"]}"' + ',\n' + '\t"fn_name": "'
    encoded_skeleton = llm.encode(skeleton)
    input_ids.extend(encoded_skeleton)
    allowed_names: List[str] = []
    for i in range(len(combined_data['defs'])):
        allowed_names.append(combined_data['defs'][i]['fn_name'])

    name_result: Tuple[str, List[int]] = get_function_name(
        llm, input_ids, vocab_buckets, allowed_names, reversed_vocab)
    function_name = name_result[0]
    input_ids: List[int] = name_result[1]
    input_ids.extend(llm.encode('",\n\t"args": {'))
    function_index: int = allowed_names.index(function_name)
    function: Dict[str, Any] = combined_data['defs'][function_index]
    
    stop_tokens = [k for k, v in reversed_vocab.items() if v in [",", "}", "\n"]]

    for arg in function['args_names']:
        input_ids.extend(llm.encode(f'"{arg}": '))
        print(arg)
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
                # ask_for_str uses numpy for full vocab search, safer/easier to keep as is for now
                ask_for_str(llm, input_ids, masks_dict, reversed_vocab)
            case 'bool':
                ask_for_bool(llm, input_ids, masks_dict, reversed_vocab)
        
    print(llm.decode(input_ids[generated_index:]))
    print("\nGeneration terminée.")

    return llm.decode(input_ids[generated_index:])
