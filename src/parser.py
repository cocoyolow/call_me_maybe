from pydantic import model_validator, BaseModel
from typing import List, Dict
from typing_extensions import Self


class FunctionCall(BaseModel):
    prompt: str


class FunctionCallsValidator(BaseModel):
    items: List[FunctionCall]


class FunctionDefinition(BaseModel):
    fn_name: str
    args_names: List[str]
    args_types: Dict[str, str]
    return_type: str

    @model_validator(mode='after')
    def check_args(self) -> Self:
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
    items: List[FunctionDefinition]
