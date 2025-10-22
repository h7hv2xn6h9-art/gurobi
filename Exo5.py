import json
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path

# ----- Load data from JSON -----
with open("data/data/lot_sizing_data.json", "r") as f:
    data = json.load(f)

name = data["name"]
H    = int(data["H"])
d    = [float(val) for val in data["demand"]]
c    = [float(val) for val in data["var_cost"]]
f    = [float(val) for val in data["setup_cost"]]
h    = [float(val) for val in data["hold_cost"]]
Qmin = float(data["Qmin"])
Qmax = float(data["Qmax"])
I0   = float(data["I0"])

# Basic validation
assert len(d) == H and len(c) == H and len(f) == H and len(h) == H
assert 0 <= Qmin <= Qmax

# ----- Build model -----
with gp.Env() as env, gp.Model(name, env=env) as model:
    # Decision variables
    x = model.addVars(H, lb=0.0, name="x")                 # production qty
    y = model.addVars(H, vtype=GRB.BINARY, name="y")       # setup (produce?)
    I = model.addVars(H, lb=0.0, name="I")                 # end-of-period inventory

    # Objective: sum_t c_t x_t + f_t y_t + h_t I_t
    model.setObjective(
        gp.quicksum(c[t]*x[t] + f[t]*y[t] + h[t]*I[t] for t in range(H)),
        GRB.MINIMIZE
    )

    # Inventory balance:
    # t=0: I0 + x0 - d0 = I0
    model.addConstr(I0 + x[0] - d[0] == I[0], name="bal_0")
    # t>=1: I_{t-1} + x_t - d_t = I_t
    for t in range(1, H):
        model.addConstr(I[t-1] + x[t] - d[t] == I[t], name=f"bal_{t}")

    # Capacity and batch size constraints
    for t in range(H):
        model.addConstr(x[t] <= Qmax * y[t], name=f"cap_max_{t}")  # at most if producing
        model.addConstr(x[t] >= Qmin * y[t], name=f"cap_min_{t}")  # at least if producing

    # Optimize
    model.optimize()

    if model.SolCount:
        # If you want strict equality, keep assert; otherwise consider a tolerance:
        # import math; assert math.isclose(model.ObjVal, 1198.5, rel_tol=1e-6, abs_tol=1e-6)
        assert model.ObjVal == 1198.5
        print(f"Total cost = {model.ObjVal:.2f}")
        for t in range(H):
            print(f"t={t:2d}: y={int(y[t].X)} x={x[t].X:.1f} I={I[t].X:.1f}")
