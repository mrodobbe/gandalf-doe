import pandas as pd
from gandalf_doe.domain import Domain, Variable
from gandalf_doe.experiment import Experiment
import numpy as np


def get_domain():
    domain = Domain()
    domain.add_variable([Variable("Temperature", "reaction_temperature", np.arange(523, 783, 10), "discrete")])
    domain.add_variable([Variable("Pressure", "reactor_pressure", np.arange(1, 11, 1), "discrete")])
    domain.add_variable([Variable("GHSV", "gas_hourly_space_velocity", np.arange(3300, 28050, 1650), "discrete")])
    domain.add_variable([Variable("Ni", "nickel_load", np.arange(0, 26, 1), "discrete")])
    domain.add_variable([Variable("Co", "cobalt_load", np.arange(0, 11, 1), "discrete")])
    domain.add_variable([Variable("Calcination", "calcination_temperature", np.arange(623, 973, 50), "discrete")])
    domain.add_variable([Variable("Reduction", "reduction_temperature", np.arange(623, 973, 50), "discrete")])
    domain.setup_space()
    return domain


def test_domain():
    domain = get_domain()
    variables = [var.name for var in domain.variables]

    assert variables == ['Temperature', 'Pressure', 'GHSV', 'Ni', 'Co', 'Calcination', 'Reduction']
    print("Success!")


def test_initialization():
    domain = get_domain()
    exp = Experiment(domain, n_init=3, mode="EMOC", clustering=True)
    sug = exp.initialize_experiments(fixed_pool=False)
    assert len(sug.index) == 3
    print("Success!")


def test_single_suggestion():
    df = pd.read_excel("test_data/dataset.xlsx", index_col=None)
    df_init = df.copy()
    domain = get_domain()
    exp = Experiment(domain, n_init=3, mode="EMOC", clustering=True)
    sug = exp.suggest_experiments(previous=df)
    assert len(sug.index) == len(df_init.index) + 1
    print("Success!")


def test_multiple_suggestions():
    df = pd.read_excel("test_data/dataset.xlsx", index_col=None)
    df_init = df.copy()
    domain = get_domain()
    exp = Experiment(domain, n_init=3, mode="EMOC", clustering=True)
    sug = exp.suggest_batch_of_experiments(previous=df, n_clusters=5, n_conditions=5)
    assert len(sug.index) > len(df_init.index)
    print("Success!")
