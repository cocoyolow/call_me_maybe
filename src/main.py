from pathlib import Path
import sys
import json
from .parser import FunctionDefinitionsValidator, FunctionCallsValidator
from .generation import start_generation
from pydantic import ValidationError
from typing import Dict


def main() -> None:
    """Start the Call Me Maybe program"""
    path: Path = Path(__file__).parent.parent / 'data' / 'output'
    path.mkdir(parents=True, exist_ok=True)
    data: Dict[str, FunctionCallsValidator | FunctionDefinitionsValidator] = {}
    try:
        json_calls_path = path.parent / 'input' / 'function_calling_tests.json'
        with open(json_calls_path, 'r') as f:
            file_info = json.load(f)
            call_validation = FunctionCallsValidator(items=file_info)
            print('function call json [OK]')
            data['calls'] = call_validation

        json_defs_path = path.parent / 'input' / 'functions_definition.json'
        with open(json_defs_path, 'r') as f:
            file_info = json.load(f)
            def_validation = FunctionDefinitionsValidator(items=file_info)
            print('function definitions json [OK]')
            data['defs'] = def_validation

        result: str = start_generation(vocab)
    except ValidationError as e:
        print("Validation error:", e)
        sys.exit(1)

    except Exception as e:
        err_type = e.__class__.__name__
        print(f'Error appened:\nError type: {err_type}\nError message: {e}')
        sys.exit(1)


if __name__ == "__main__":
    main()
