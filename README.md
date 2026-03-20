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
Or run it manually with custom files:
```bash
uv run python -m src --functions_definition <path_to_defs.json> --input <path_to_inputs.json> --output <path_to_output.json>
```

## Resources
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Qwen Model Documentation](https://huggingface.co/Qwen)
- **AI Usage:** AI assistants (such as ChatGPT and Gemini) were used to assist with debugging execution issues, refining regular expressions, code formatting (PEP 257 docstrings), and structuring the constrained decoding logic and state machines.

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
# Given a function definition for 'get_weather(location: string)'
# and the user prompt "What is the weather like in Paris?"

$ uv run python -m src
function call json [OK]
function definitions json [OK]
prompt 1/1: "What is the weather like in Paris?"
	   - > detected function: get_weather
	   - > location (string): Paris

Generation finished.
```
