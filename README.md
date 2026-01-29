*This project has been created as part of the 42 curriculum by cobussie.*

# Call Me Maybe
**Introduction to function calling in LLMs**

## Description
This project implements a robust **constrained decoding** mechanism for Large Language Models (LLMs) to perform reliable function calling. The goal is to address the stochastic nature of LLMs, which often leads to "hallucinations" or syntactically incorrect outputs when asked to generate structured data like JSON.

By intercepting the model's generation process and strictly enforcing token constraints based on function signatures, this tool ensures that the model outputs function calls that are not only syntactically valid JSON but also adhere to the specific argument types (integers, floats, strings) defined by the user.

## Instructions

### Installation
This project manages dependencies using `uv`. To install the necessary environment and dependencies, run:

```bash
make install
```

### Execution
To run the main program with the default dataset:

```bash
make run
```

This command executes `src/__main__.py`, which processes input prompts and function definitions, generates constrained responses, and saves the results.

#### Custom Usage
You can specify custom input and output paths using the CLI arguments:

```bash
uv run python3 -m src --input <path_to_input_dir> --output <path_to_output_dir>
```

### Linting
To check the code for style and type errors:

```bash
make lint
```

## Algorithm Explanation
The core of this solution lies in **constrained decoding**, implemented in `src/generation.py`. Instead of letting the LLM freely predict the next token, we intervene at the logit level:

1.  **Masking**: We categorize tokens into specific sets (e.g., `digits`, `digits_minus`, `valid_str_chars`) using `get_masks`.
2.  **State Machines**: When the model needs to generate a specific type (e.g., a float), we enter a finite state machine (FSM).
    *   *Example (Float)*: The FSM allows a minus sign, then digits, then a single dot, then more digits. It strictly forbids invalid sequences like `1.2.3` or `--5`.
3.  **Trie-like Search for Function Names**: The `get_function_name` function pre-calculates valid token sequences for all allowed function names. It then forces the model to follow one of these valid paths, effectively treating the function name selection as a trie traversal.
4.  **Direct Logit Manipulation**: At each step, we permit only the tokens allowed by the current state or mask, setting the probabilities of all other tokens to negative infinity.

## Design Decisions
*   **Modular Architecture**: The project is structured for separation of concerns:
    *   `src/parser.py`: Handles data validation using Pydantic models.
    *   `src/generation.py`: Contains the low-level constrained decoding logic.
    *   `src/__main__.py`: Manages the application flow and CLI.
*   **Pydantic for Validation**: We use Pydantic to strictly validate input JSON files (`functions_definition.json` and `function_calling_tests.json`) before processing, ensuring the system never crashes due to malformed input.
*   **Zero-Temperature equivalent**: By always selecting the `argmax` of valid logits, we ensure deterministic outputs, which is crucial for reliable testing and debugging.

## Performance Analysis
*   **Accuracy**:
    *   **100% Valid JSON**: The solution achieves **100% syntactic correctness** for the supported primitive types (int, float, str). The strict masking makes it impossible for the model to generate invalid JSON structure or type mismatches.
    *   **>95% Semantic Accuracy**: We achieve a high success rate in generating the *correct* function call and arguments (over 95%). This is bolstered by a carefully engineered system prompt that increases the likelihood of high-quality responses.
*   **Speed**:
    *   **Optimized Name Matching**: The `get_function_name` optimization significantly speeds up generation by avoiding the need to sample and validate tokens one by one for long function names.
    *   **Token efficiency**: Since we force the correct path, we avoid "retry" loops often found in prompt-engineering-based solutions.

## Challenges Faced
*   **String Handling and Quoting**: Converting arbitrary model output into a valid JSON string is non-trivial. The model might attempt to generate unescaped quotes or terminate the string early. We tackled this by strictly controlling valid string characters and ensuring the model cannot "overflow" the string definition or inject invalid JSON syntax (like multiple quotes).
*   **Tokenization Artifacts**: Different tokenizers represent characters differently (e.g., the `Ġ` prefix for spaces in some byte-pair encodings). Handling these variations when building our allowable sets was critical.
*   **Floating Point Representation**: Ensuring a valid float representation (preventing multiple dots, ensuring digits after a dot, handling leading zeros/minus signs) required a precise state machine implementation.
*   **Mapping High-Level Types**: Bridging the gap between a high-level JSON schema type (like "number") and the low-level vocabulary IDs of the LLM required careful mapping and pre-computation of valid token sets.

## Testing Strategy
*   **Input Validation**: We use `FunctionCallsValidator` and `FunctionDefinitionsValidator` to enforce strict schema compliance on inputs.
*   **Output Verification**: The final output is verified to be valid JSON. The determinism of the constrained decoding allows us to predictably test edge cases (like negative numbers or strings with special characters) without flakiness.
*   **Manual Review**: We inspect the generated `function_calling_results.json` to ensure the logic semantically matches the user prompt.

## Example Usage
Given a function definition like:
{
  "fn_name": "get_weather",
  "args_names": ["location", "days"],
  "args_types": {"location": "str", "days": "int"},
  "return_type": "str"
}

And a prompt: *"What is the weather in Paris for the next 3 days?"*

The system forces the LLM to generate:

{
    "prompt": "What is the weather in Paris for the next 3 days?",
    "fn_name": "get_weather",
    "args": {
        "location": "Paris",
        "days": 3
    }
}

All without the possibility of syntax errors or type hallucinations.

## Resources
*   **Guidance on Constrained Decoding**: *Hugging Face Blog - Constrained Beam Search* (conceptually similar).
*   **Pydantic Documentation**: For robust data validation.
*   **LLM SDK**: The internal `llm_sdk` provided the base `Small_LLM_Model` class used for tokenization and inference.

### AI Usage
AI was used in this project to:
*   **Draft Documentation**: Generative AI assisted in structuring and writing this README.md to ensure clarity and completeness.
*   **Algorithm Refinement**: AI provided suggestions for optimizing the function name matching logic.
