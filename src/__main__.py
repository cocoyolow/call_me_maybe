from pathlib import Path
import sys
import json
from .parser import FunctionDefinitionsValidator, FunctionCallsValidator
from .generation import start_generation
from pydantic import ValidationError
from typing import Dict, List, Any, Tuple


def exit_wrong_format() -> None:
    """Prints an error message telling the user how to use the program,
    and exits the program

    Args:
        None

    Returns:
        None
    """
    print('error: wrong format: the correct format is :\n',
          'uv run python -m src',
          '[--input <input_file>] [--output <output_file>]')
    sys.exit(1)


def validate_args(argv: List[str], argc: int) -> Tuple[Path, Path]:
    """Validates the arguments passed to the program and returns the two paths:
    input and output

    Args:
        argv (List[str]): the arguments passed to the program
        argc (int): the number of arguments passed to the program

    Returns:
        Path, Path: the two paths
    """
    path_input: Path
    path_output: Path

    if argc == 2 or argc == 4:
        exit_wrong_format()
    if argc == 1:
        path_input = Path(__file__).parent.parent / 'data' / 'input'
        path_output = Path(__file__).parent.parent / 'data' / 'output'

    elif argc == 3:
        if sys.argv[1] == '--input':
            path_input = Path(sys.argv[2])
        elif sys.argv[1] == '--output':
            path_output = Path(sys.argv[2])
        else:
            exit_wrong_format()
    elif argc == 5:
        if sys.argv[1] == '--input':
            path_input = Path(sys.argv[2])
        else:
            exit_wrong_format()
        if sys.argv[3] == '--output':
            path_output = Path(sys.argv[4])
        else:
            exit_wrong_format()
    return path_input, path_output


def main() -> None:
    """Start the Call Me Maybe program

    Args:
        None

    Returns:
        None
    """

    argc: int = len(sys.argv)

    path_input: Path
    path_output: Path
    path_input, path_output = validate_args(sys.argv, argc)

    data: Dict[str, List[Dict[str, Any]]] = {}
    try:
        json_calls_path = path_input / 'function_calling_tests.json'
        with open(json_calls_path, 'r') as f:
            file_info = json.load(f)
            FunctionCallsValidator(items=file_info)
            print('function call json [OK]')
            data['calls'] = file_info

        json_defs_path = path_input / 'functions_definition.json'
        with open(json_defs_path, 'r') as f:
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
        path_output.mkdir(parents=True, exist_ok=True)
        result: List[Dict[str, Any]] = start_generation(data)
        with open(path_output / "function_calling_results.json", "w") as f:
            json.dump(result, f, indent=4)

    except Exception as e:
        err_type = e.__class__.__name__
        print(f'Error happened:\nError type: {err_type}\nError message: {e}')
        sys.exit(1)


if __name__ == "__main__":
    main()
