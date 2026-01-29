from pydantic import model_validator, BaseModel
from typing import List, Dict
from typing_extensions import Self


class FunctionCall(BaseModel):
    """Function call class, used to validate
    the function calls in the json file"""
    prompt: str


class FunctionCallsValidator(BaseModel):
    """class to check a list of function calls, used to validate
    all the function calls in the json file"""
    items: List[FunctionCall]


class FunctionDefinition(BaseModel):
    """Function definition class, used to validate
    the function definitions in the json file.

    Attributes:
        fn_name (str): The name of the function.
        args_names (List[str]): A list of argument names.
        args_types (Dict[str, str]): A dictionary mapping argument
        names to their types.
        return_type (str): The return type of the function.
    """
    fn_name: str
    args_names: List[str]
    args_types: Dict[str, str]
    return_type: str

    @model_validator(mode='after')
    def check_args(self) -> Self:
        """Checks if the function definition is valid

        Returns:
            Self: the instance calling the function
        """
        names_set = set(self.args_names)
        types_set = set(self.args_types.keys())

        if len(names_set) != len(self.args_names):
            raise ValueError(
                f"Function '{self.fn_name}': Duplicate argument names found.")
        if len(types_set) != len(self.args_types):
            raise ValueError(
                f"Function '{self.fn_name}': Duplicate argument types found.")
        if names_set != types_set:
            raise ValueError(
                f"Function '{self.fn_name}': 'args_names' and 'args_types'",
                "'does not correspond to each other'. "
                f"Defined args: {self.args_names},",
                f"Defined types keys: {list(self.args_types.keys())}"
            )
        return self


class FunctionDefinitionsValidator(BaseModel):
    """class the check a list of function definitions, used to validate
    all the function definitions in the json file"""
    items: List[FunctionDefinition]
