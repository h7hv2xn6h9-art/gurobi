import numpy as np
import gurobipy as gp
from gurobipy import GRB

def generate_knapsack(num_items):
    # Fix seed value
    rng = np.random.default_rng(seed=0)
    # Item values, weights
    values = rng.uniform(low=1, high=25, size=num_items)
    weights = rng.uniform(low=5, high=100, size=num_items)
    # Knapsack capacity
    capacity = 0.7 * weights.sum()
    return values, weights, capacity

def solve_knapsack_model(values, weights, capacity):
    num_items = len(values)

    # Indices
    I = range(num_items)

    # Turn numpy arrays into tupledicts (same keys as decision vars)
    val = gp.tupledict({i: float(values[i]) for i in I})
    wt  = gp.tupledict({i: float(weights[i]) for i in I})

    with gp.Env() as env:
        with gp.Model(name="knapsack", env=env) as model:
            # Optional: quieter logs for big instances
            model.Params.OutputFlag = 1  # set to 0 to silence

            # Decision variables: x[i] in {0,1}
            x = model.addVars(I, vtype=GRB.BINARY, name="x")

            # Objective: maximize sum r_i * x_i
            model.setObjective(val.prod(x), GRB.MAXIMIZE)

            # Capacity constraint: sum c_i * x_i <= C
            model.addConstr(wt.prod(x) <= capacity, name="capacity")

            model.optimize()

            # Extract solution
            if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
                selected = [i for i in I if x[i].X > 0.5]
                total_value = sum(values[i] for i in selected)
                total_weight = sum(weights[i] for i in selected)
                print(f"Selected items: {len(selected)}")
                print(f"Total value: {total_value:.4f}")
                print(f"Total weight: {total_weight:.4f} / {capacity:.4f}")
                return selected, total_value, total_weight
            else:
                print(f"Model ended with status {model.Status}")
                return [], 0.0, 0.0

# Example run
values, weights, capacity = generate_knapsack(10000)
solve_knapsack_model(values, weights, capacity)
