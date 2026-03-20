*This project has been created as part of the 42 curriculum by cobussie.*

# Call Me Maybe

## Description
**Call Me Maybe** is an introduction to function calling in Large Language Models (LLMs). The goal of this project is to implement constrained decoding techniques to enforce an LLM to generate precise JSON outputs corresponding to predefined function signatures. It uses a lightweight local model (`Qwen/Qwen3-0.6B`) and ensures that the model only produces valid function names and correctly typed arguments (strings, integers, floats) according to the provided system prompts.

## Instructions
### Prerequisites
- Python 3.10+
- `uv` package manager
- `make`

### Installation
Clone the repository and install the dependencies using the provided Makefile:
```bash
git clone <repository_url>
cd call_me_maybe
make install
```
This will run `uv sync` to set up the virtual environment and install all dependencies.

### Execution
To run the program with the default input and output files:
```bash
make run
```

### CLI options
The CLI entrypoint is `python -m src` and supports 3 optional arguments:

| Option | Type | Default value | Description |
|---|---|---|---|
| `--functions_definition` | path | `data/input/functions_definition.json` | Path to the function definitions JSON file |
| `--input` | path | `data/input/function_calling_tests.json` | Path to the prompts/calls JSON file |
| `--output` | path | `data/output/function_calling_results.json` | Path to the generated output JSON file |

Example with all options:
```bash
uv run python -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calling_results.json
```

### `main()` parameters (programmatic usage)
The function `main()` in `src/__main__.py` has this signature:

```python
def main(llm_name: str | None = None, tests: int = 0) -> None
```

- `llm_name`: optional model name passed to `Small_LLM_Model(llm_name)`. If `None`, the SDK default model is used.
- `tests`: if `tests != 0`, the input prompts loaded from `--input` are replaced by `tests` synthetic prompts generated with Faker.

Important note: when running via CLI (`python -m src`), the module currently calls:

```python
if __name__ == "__main__":
    main()
```

So from CLI, `tests` stays at `0` and the program uses prompts from the input JSON file.

Also note that `llm_name` and `tests` are not exposed as CLI flags in the current implementation.

Programmatic example:

```python
from src.__main__ import main

# Use another model and limit the run to 5 prompts
main(llm_name="Qwen/Qwen3-0.6B", tests=5)
```

### Input/Output JSON format
Expected input files:

1. `functions_definition.json`: list of function definitions.
2. `function_calling_tests.json`: list of prompts to resolve.

Minimal examples:

```json
[
	{
		"name": "fn_add_numbers",
		"description": "Add two numbers together and return their sum.",
		"parameters": {
			"a": {"type": "number"},
			"b": {"type": "number"}
		},
		"returns": {"type": "number"}
	}
]
```

```json
[
	{"prompt": "What is the sum of 2 and 3?"}
]
```

Supported parameter types in generation are: `string`, `integer`, `number`.

### Useful Make targets
- `make install`: install dependencies (`uv sync`)
- `make run`: run the program with default files
- `make debug`: launch with `pdb`
- `make lint`: run `flake8` and `mypy`
- `make clean` / `make fclean`: remove caches and generated files

## Resources
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Qwen Model Documentation](https://huggingface.co/Qwen)
- **AI Usage:** AI assistants (such as ChatGPT and Gemini) were used to Create README.md, and for the code formatting (PEP 257 docstrings).

## Algorithm explanation
The project relies on **constrained decoding** to force the LLM into generating syntactically correct text that strictly adheres to the requested function signatures:
- **Function Name Generation:** Instead of letting the model free-generate text, we pre-calculate valid token sequences for all allowed function names. During generation, we restrict the model's logits to only allow tokens that progress toward one of the valid function names.
- **Parameter Value Generation:** We dynamically apply masks (allow-lists of token IDs) based on the parameter type:
  - For `integer`, only digits and a minus sign are allowed initially, followed by stop tokens (like commas or braces).
  - For `number` (float), we use a state machine (`START`, `INT_PART`, `AFTER_DOT`, `DECIMAL_PART`) to ensure that the generated output strictly forms a valid floating-point number.
  - For `string`, we restrict the tokens to a set of valid string characters and stop at the closing quote.

## Design decisions
- **Pydantic Validation:** Used `pydantic` to rigorously validate input function definitions and input queries, ensuring the JSON data is structurally sound before passing it to the LLM.
- **Token Masking:** By passing valid subsets of the vocabulary to `numpy.argmax`, we override the LLM's natural tendency to generate conversational padding.
- **State Machines:** Managing state during numerical decoding ensures that malformed numbers (like `12.34.56`) cannot be generated.
- **uv over pip:** `uv` is used for extremely fast dependency management and reproducible environments, keeping the runtime execution deterministic.

## Performance analysis
- **Accuracy:** Due to the strict constrained decoding approach, the structural accuracy of the generated JSON is virtually 100%. The model mathematically cannot output an invalid function name or malformed numbers/strings.
- **Speed:** Limiting the vocabulary mask at each step introduces a slight CPU overhead during the logits evaluation, but prevents the model from generating long, unnecessary conversational outputs, drastically improving the overall token-generation speed compared to unconstrained completion.
- **Reliability:** By handling exceptions and invalid JSON structures safely, the pipeline ensures uninterrupted batch processing without crashing midway through generation.

## Challenges faced
- **Tokenization Quirks:** Handling subword tokenization meant that some characters might be merged, or spaces might be prepended. Reconstructing exact strings and ensuring digits or special characters are accurately parsed required building inverse vocabularies and careful token matching.
- **Mypy and Local Modules:** Dealing with static typing issues related to a shadowed local `llm_sdk` directory required bypassing static analyzer limitations while keeping runtime imports fully functional.
- **State Management:** Tracking the exact state of float generation (e.g., distinguishing between negative signs and decimal points) required precise logic to avoid infinite generation loops or illegal formats.

## Testing strategy
The implementation is validated through a combination of:
1. **Pydantic Validation:** Every input parsed is validated for strictly matching the expected schema.
2. **Automated Linting:** Code is tested against strict typing using `mypy` and PEP 8 correctness using `flake8` (`make lint`).
3. **End-to-End Tests:** We pass a bulk set of varied mock prompts and evaluate the valid output JSON format to ensure the constrained decoding never breaks under various edge cases, including empty strings, large numbers, special characters, wrong types, ambiguous prompts, and functions with multiple parameters.

## Example usage
```bash
# Given a function definition for 'get_power(a: number, b: number)'
# and the user prompt "What is the power of 2 to the power of 3?"

$ uv run python -m src
function call json [OK]
function definitions json [OK]
prompt 1/1: "What is the power of 2 to the power of 3?"
	   - > detected function: get_power
	   - > a (number): 2.0
	   - > b (number): 3.0

Generation finished.


# Given a function definition for 'fn_greet(name: string)'
# and the user prompt "Greet shrek"

$ uv run python -m src
function call json [OK]
function definitions json [OK]
prompt 1/1: "Greet shrek"
	   - > detected function: fn_greet
	   - > name (string): shrek

Generation finished.
```
