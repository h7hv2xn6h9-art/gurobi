import gurobipy as gp
from gurobipy import GRB

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
T = range(len(load_forecast))
required_power = {t: load_forecast[t] - solar_forecast[t] for t in T}

thermal_units = ["gen1", "gen2", "gen3"]

# couts: a + b*p + c*p^2, startup, shutdown
_, a, b, c, sup_cost, sdn_cost = gp.multidict(
    {
        "gen1": [5.0, 0.5, 1.0, 2, 1],
        "gen2": [5.0, 0.5, 0.5, 2, 1],
        "gen3": [5.0, 3.0, 2.0, 2, 1],
    }
)
# limites
_, pmin, pmax = gp.multidict(
    {"gen1": [1.5, 5.0], "gen2": [2.5, 10.0], "gen3": [1.0, 3.0]}
)
# statut initial
_, init_status = gp.multidict({"gen1":[0], "gen2":[0], "gen3":[0]})

# ------------------ Modele sans SciPy ------------------
with gp.Env() as env, gp.Model("UC_no_scipy", env=env) as model:
    # Variables indexees
    p  = model.addVars(thermal_units, T, lb=0.0, name="p")                 # puissance
    u  = model.addVars(thermal_units, T, vtype=GRB.BINARY, name="u")       # on/off
    v  = model.addVars(thermal_units, T, vtype=GRB.BINARY, name="v")       # startup
    w  = model.addVars(thermal_units, T, vtype=GRB.BINARY, name="w")       # shutdown

    # Objectif: sum_{i,t} [ c_i p^2 + b_i p + a_i u + sup_i v + sdn_i w ]
    obj = gp.quicksum(
        c[g] * (p[g,t] * p[g,t]) + b[g] * p[g,t] + a[g] * u[g,t]
        + sup_cost[g] * v[g,t] + sdn_cost[g] * w[g,t]
        for g in thermal_units for t in T
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # Bilan de puissance
    model.addConstrs(
        (gp.quicksum(p[g,t] for g in thermal_units) == required_power[t] for t in T),
        name="power_balance"
    )

    # Logique on/off
    for g in thermal_units:
        # t=0
        model.addConstr(u[g,0] - init_status[g] == v[g,0] - w[g,0], name=f"logical_init[{g},0]")
        # t>0
        for t in T:
            if t == 0: 
                continue
            model.addConstr(u[g,t] - u[g,t-1] == v[g,t] - w[g,t], name=f"logical_trans[{g},{t}]")
        # Pas start et stop simultanes
        model.addConstrs((v[g,t] + w[g,t] <= 1 for t in T), name=f"no_simul[{g}]")

    # Contraintes physiques via indicatrices
    for g in thermal_units:
        for t in T:
            model.addGenConstrIndicator(u[g,t], True,  p[g,t] >= pmin[g], name=f"pmin[{g},{t}]")
            model.addGenConstrIndicator(u[g,t], True,  p[g,t] <= pmax[g], name=f"pmax[{g},{t}]")
            model.addGenConstrIndicator(u[g,t], False, p[g,t] == 0.0,     name=f"poff[{g},{t}]")

    model.optimize()

    # Affichage
    if model.SolCount:
        print(f"OverAll Cost = {model.ObjVal:.2f}\n")
        print("%5s" % "time", end=" ")
        for t in T: print("%4d" % t, end=" ")
        print()
        for g in thermal_units:
            print("%5s" % g, end=" ")
            for t in T: print("%4.1f" % p[g,t].X, end=" ")
            print()
        print("%5s" % "Solar", end=" ")
        for t in T: print("%4.1f" % solar_forecast[t], end=" ")
        print("\n%5s" % "Load", end=" ")
        for t in T: print("%4.1f" % load_forecast[t], end=" ")
        print()
