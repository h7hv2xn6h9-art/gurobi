import gurobipy as gp
from gurobipy import GRB
import numpy as np

# ------------------ Donnees ------------------
load_forecast = [
     4,  4,  4,  4,  4,  4,   6,   6,
    12, 12, 12, 12, 12,  4,   4,   4,
     4, 16, 16, 16, 16,  6.5, 6.5, 6.5,
]
solar_forecast = [
    0,   0,   0,   0,   0,   0,   0.5, 1.0,
    1.5, 2.0, 2.5, 3.5, 3.5, 2.5, 2.0, 1.5,
    1.0, 0.5, 0,   0,   0,   0,   0,   0,
]
T = len(load_forecast)
required_power = [l - s for l, s in zip(load_forecast, solar_forecast)]

thermal_units = ["gen1", "gen2", "gen3"]

# costs: a + b*p + c*p^2, startup, shutdown
_, a, b, c, sup_cost, sdn_cost = gp.multidict(
    {"gen1":[5.0, 0.5, 1.0, 2, 1],
     "gen2":[5.0, 0.5, 0.5, 2, 1],
     "gen3":[5.0, 3.0, 2.0, 2, 1]}
)
_, pmin, pmax = gp.multidict(
    {"gen1":[1.5, 5.0], "gen2":[2.5, 10.0], "gen3":[1.0, 3.0]}
)
_, init_status = gp.multidict({"gen1":[0], "gen2":[0], "gen3":[0]})

G = range(len(thermal_units))
TT = range(T)

def show_results(model, p, y):
    print(f"OverAll Cost = {model.ObjVal:.2f}\n")
    print("%5s" % "time", end=" ")
    for t in TT:
        print("%4s" % t, end=" ")
    print("\n")
    for gi, g in enumerate(thermal_units):
        print("%5s" % g, end=" ")
        for t in TT:
            print("%4.1f" % p[gi,t].X, end=" ")
        print("\n")
    print("%5s" % "Solar", end=" ")
    for t in TT:
        print("%4.1f" % solar_forecast[t], end=" ")
    print("\n")
    print("%5s" % "Load", end=" ")
    for t in TT:
        print("%4.1f" % load_forecast[t], end=" ")
    print("\n")

# ------------------ Modele sans SciPy ------------------
with gp.Env() as env, gp.Model(env=env) as model:
    # Variables indexees (pas de MVar)
    p  = model.addVars(G, TT, lb=0.0, name="p")                 # puissance
    y  = model.addVars(G, TT, vtype=GRB.BINARY, name="y")       # commitment
    su = model.addVars(G, TT, vtype=GRB.BINARY, name="su")      # startup
    sd = model.addVars(G, TT, vtype=GRB.BINARY, name="sd")      # shutdown

    # Objectif: somme de (a*y + b*p + c*p^2 + sup_cost*su + sdn_cost*sd)
    obj = gp.quicksum(
        a[thermal_units[gi]] * y[gi,t] +
        b[thermal_units[gi]] * p[gi,t] +
        c[thermal_units[gi]] * (p[gi,t] * p[gi,t]) +
        sup_cost[thermal_units[gi]] * su[gi,t] +
        sdn_cost[thermal_units[gi]] * sd[gi,t]
        for gi in G for t in TT
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # Equilibre puissance pour chaque periode
    model.addConstrs(
        (gp.quicksum(p[gi,t] for gi in G) == required_power[t] for t in TT),
        name="power_balance"
    )

    # Logique de commitment
    # t = 0
    model.addConstrs(
        (y[gi,0] - init_status[thermal_units[gi]] == su[gi,0] - sd[gi,0] for gi in G),
        name="logical_initial"
    )
    # t > 0
    model.addConstrs(
        (y[gi,t] - y[gi,t-1] == su[gi,t] - sd[gi,t] for gi in G for t in range(1, T)),
        name="logical_transitions"
    )
    # Pas startup et shutdown en meme temps
    model.addConstrs((su[gi,t] + sd[gi,t] <= 1 for gi in G for t in TT), name="no_simul")

    # Contraintes physiques via indicateurs
    for gi in G:
        gname = thermal_units[gi]
        for t in TT:
            model.addGenConstrIndicator(y[gi,t], True,  p[gi,t] >= pmin[gname], name=f"pmin[{gname},{t}]")
            model.addGenConstrIndicator(y[gi,t], True,  p[gi,t] <= pmax[gname], name=f"pmax[{gname},{t}]")
            model.addGenConstrIndicator(y[gi,t], False, p[gi,t] == 0.0,          name=f"poff[{gname},{t}]")

    model.optimize()
    if model.SolCount:
        show_results(model, p, y)
