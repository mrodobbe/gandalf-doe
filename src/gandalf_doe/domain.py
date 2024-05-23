import GPyOpt
import numpy as np
from typing import Union, List, Any
from numpy.typing import NDArray


class Variable:

    def __init__(self,
                 name: str,
                 desc: str,
                 domain: Union[List[Any], NDArray[Any]],
                 var_type: str):

        self.name = name
        self.description = desc
        self.domain = domain
        self.var_type = var_type


class Constraint:

    def __init__(self,
                 name: str,
                 constraint: str):

        self.name = name
        self.constraint = constraint


class Domain:

    def __init__(self,
                 variables: Union[None, List[Variable]] = None,
                 constraints: Union[None, List[Constraint]] = None):

        if constraints is None:
            constraints = []
        if variables is None:
            variables = []

        self.variables = variables
        self.constraints = constraints
        self.design_space = []
        self.min_values: Union[List[Any], NDArray[Any]] = []
        self.max_values: Union[List[Any], NDArray[Any]] = []

    def add_variable(self, variable: Union[List[Variable], Variable]) -> None:
        self.variables += variable

    def add_constraint(self, constraint: Union[List[Constraint], Constraint]) -> None:
        self.constraints += constraint

    def setup_space(self) -> None:
        input_space = []
        constraint_space = []
        if type(self.min_values) != list:
            self.min_values = list(self.min_values)
            self.max_values = list(self.max_values)
        for variable in self.variables:
            input_space.append({"name": variable.name,
                                "type": variable.var_type,
                                "domain": tuple(variable.domain)})
            self.min_values.append(variable.domain[0])
            self.max_values.append(variable.domain[-1])

        self.min_values = np.asarray(self.min_values).astype(np.float64)
        self.max_values = np.asarray(self.max_values).astype(np.float64)

        for constraint in self.constraints:
            constraint_space.append({"name": constraint.name,
                                     "constraint": constraint.constraint})
        if len(self.constraints) == 0:
            self.design_space = GPyOpt.Design_space(space=input_space)
        else:
            self.design_space = GPyOpt.Design_space(space=input_space, constraints=constraint_space)
