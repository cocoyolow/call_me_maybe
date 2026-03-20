from pydantic import BaseModel
from typing import List, Dict


class FunctionCall(BaseModel):
    """Function call class, used to validate
    the function calls in the json file"""
    prompt: str


class FunctionCallsValidator(BaseModel):
    """class to check a list of function calls, used to validate
    all the function calls in the json file"""
    items: List[FunctionCall]


class ParameterDetail(BaseModel):
    """Class to validate the details of a specific parameter"""
    type: str


class ReturnDetail(BaseModel):
    """Class to validate the return details of a function"""
    type: str


class FunctionDefinition(BaseModel):
    """Function definition class, used to validate
    the function definitions in the json file.

    Attributes:
        name (str): The name of the function.
        description (str): The description of the function.
        parameters (Dict[str, ParameterDetail]): A dictionary mapping argument
        names to their details (like type).
        returns (ReturnDetail): The return type details of the function.
    """
    name: str
    description: str
    parameters: Dict[str, ParameterDetail] = {}
    returns: ReturnDetail


class FunctionDefinitionsValidator(BaseModel):
    """class the check a list of function definitions, used to validate
    all the function definitions in the json file"""
    items: List[FunctionDefinition]
