from pathlib import Path
from typing import List, Dict
from pydantic import TypeAdapter
import sys
import json

# from llm_sdk import Small_LLM_Model


def main() -> None:
    assert __file__ is not None
    path: Path = Path(__file__).parent.parent / 'data' / 'output'
    path.mkdir(parents=True, exist_ok=True)
    try:
        validator: TypeAdapter = TypeAdapter(List[Dict[str, str]])
        json_path = path.parent / 'input' / 'functions_definition.json'
        with open(json_path, 'r') as f:
            data = json.load(f)
            print(f'Data: {data}')
            # validator.validate_python(data)
    except Exception as e:
        err_type = e.__class__.__name__
        print(f'Error appened:\nError type: {err_type}\nError message: {e}')
        sys.exit(1)


if __name__ == "__main__":
    main()
