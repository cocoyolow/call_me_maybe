from pathlib import Path
import sys
import json
from .parser import FunctionDefinitionsValidator, FunctionCallsValidator
from .generation import start_generation
from pydantic import ValidationError
from typing import Dict, List, Any


def main() -> None:
    """Start the Call Me Maybe program"""

    argc: int = len(sys.argv)

    path_input: Path
    path_output: Path
    if argc == 1:
        path_input = Path(__file__).parent.parent / 'data' / 'input'
        path_output = Path(__file__).parent.parent / 'data' / 'output'
    elif argc == 2:
        if sys.argv[1] == '--input':
            path_input = sys.argv[2]
        elif sys.argv[1] == '--output':
            path_output = sys.argv[2]

    path: Path = Path(__file__).parent.parent / 'data' / 'output'
    path.mkdir(parents=True, exist_ok=True)
    data: Dict[str, List[Dict[str, Any]]] = {}
    try:
        json_calls_path = path.parent / 'input' / 'function_calling_tests.json'
        with open(json_calls_path, 'r') as f:
            file_info = json.load(f)
            FunctionCallsValidator(items=file_info)
            print('function call json [OK]')
            data['calls'] = file_info

        json_defs_path = path.parent / 'input' / 'functions_definition.json'
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
        print(f'Error appened:\nError type: {err_type}\nError message: {e}')
        sys.exit(1)

    try:
        result: List[Dict[str, Any]] = start_generation(data)
        with open(path / "function_calling_results.json", "w") as f:
            json.dump(result, f, indent=4)
    except Exception as e:
        err_type = e.__class__.__name__
        print(f'Error appened:\nError type: {err_type}\nError message: {e}')
        sys.exit(1)


if __name__ == "__main__":
    main()
