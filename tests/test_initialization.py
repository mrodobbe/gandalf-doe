from gandalf_doe.experiment import Experiment
from gandalf_doe.domain import Domain, Variable
import numpy as np


def test_initialization():
    domain = Domain()
    domain.add_variable([Variable("Temperature", "reaction_temperature", np.arange(523, 783, 10), "discrete")])
    domain.add_variable([Variable("Pressure", "reactor_pressure", np.arange(1, 11, 1), "discrete")])
    domain.add_variable([Variable("GHSV", "gas_hourly_space_velocity", np.arange(3300, 28050, 1650), "discrete")])
    domain.add_variable([Variable("Ni", "nickel_load", np.arange(0, 26, 1), "discrete")])
    domain.add_variable([Variable("Co", "cobalt_load", np.arange(0, 11, 1), "discrete")])
    domain.add_variable([Variable("Calcination", "calcination_temperature", np.arange(623, 973, 50), "discrete")])
    domain.add_variable([Variable("Reduction", "reduction_temperature", np.arange(623, 973, 50), "discrete")])
    domain.setup_space()
    variables = [var.name for var in domain.variables]

    exp = Experiment(domain, n_init=3, mode="EMOC", clustering=True)
    sug = exp.initialize_experiments(fixed_pool=False)
    assert len(sug.index) == 3
    print("Success!")
