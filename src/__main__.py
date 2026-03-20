from pathlib import Path
import sys
import json
from .parser import FunctionDefinitionsValidator, FunctionCallsValidator
from .generation import start_generation
from pydantic import ValidationError
from typing import Dict, List, Any, Tuple


def exit_wrong_format() -> None:
    """Print an error message telling the user how to use the program and exit.

    Returns:
        None
    """
    print('error: wrong format: the correct format is :\n',
          'uv run python -m src',
          '[--input <input_file>] [--output <output_file>]')
    sys.exit(1)


def validate_args() -> Tuple[Path, Path, Path]:
    """Validate and parse the arguments passed to the program.

    Uses argparse to handle optional arguments in any order.

    Returns:
        Tuple[Path, Path, Path]: The paths for functions_definition, input, and
            output.
    """
    import argparse
    base_dir = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        prog='uv run python -m src',
        description='Call Me Maybe program',
        allow_abbrev=False
    )
    parser.add_argument(
        '--functions_definition',
        type=Path,
        default=base_dir / 'data' / 'input' / 'functions_definition.json',
        help='Path to the functions definition JSON file'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=base_dir / 'data' / 'input' / 'function_calling_tests.json',
        help='Path to the input JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=base_dir / 'data' / 'output' / 'function_calling_results.json',
        help='Path to the output JSON file'
    )
    args = parser.parse_args()

    return args.functions_definition, args.input, args.output


def main(llm_name: str | None = None, tests: int = 0) -> None:
    """Start the Call Me Maybe program.

    Args:
        llm_name (str | None): The name of the LLM to use. Defaults to None.
        tests (int): The number of tests to run. Defaults to 0.

    Returns:
        None
    """

    path_input: Path
    path_output: Path
    path_definitions: Path
    path_definitions, \
        path_input, \
        path_output = validate_args()

    data: Dict[str, List[Dict[str, Any]]] = {}
    try:
        with open(path_input, 'r') as f:
            file_info = json.load(f)
            FunctionCallsValidator(items=file_info)
            print('function call json [OK]')
            data['calls'] = file_info

        with open(path_definitions, 'r') as f:
            file_info = json.load(f)
            FunctionDefinitionsValidator(items=file_info)
            print('function definitions json [OK]')
            data['defs'] = file_info

    except ValidationError as e:
        print("Validation error:", e)
        sys.exit(1)

    except Exception as e:
        err_type = e.__class__.__name__
        print(f'Error happened:\nError type: {err_type}\nError message: {e}')
        sys.exit(1)

    try:
        path_output.parent.mkdir(parents=True, exist_ok=True)
        result: List[Dict[str, Any]] = start_generation(data, llm_name, tests)
        with open(path_output, "w") as f:
            json.dump(result, f, indent=4)

    except Exception as e:
        err_type = e.__class__.__name__
        print(f'Error happened:\nError type: {err_type}\nError message: {e}')
        sys.exit(1)


if __name__ == "__main__":
    main()
