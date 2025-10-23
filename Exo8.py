import gurobipy as gp
from gurobipy import GRB

# 24 Hour Load Forecast (MW)
load_forecast = [
     4,  4,  4,  4,  4,  4,   6,   6,
    12, 12, 12, 12, 12,  4,   4,   4,
     4, 16, 16, 16, 16,  6.5, 6.5, 6.5,
]

# solar energy forecast (MW)
solar_forecast = [
    0,   0,   0,   0,   0,   0,   0.5, 1.0,
    1.5, 2.0, 2.5, 3.5, 3.5, 2.5, 2.0, 1.5,
    1.0, 0.5, 0,   0,   0,   0,   0,   0,
]

nTimeIntervals = len(load_forecast)
thermal_units = ["gen1", "gen2", "gen3"]

# costs: a + b*p + c*p^2, startup, shutdown
thermal_units_cost, a, b, c, sup_cost, sdn_cost = gp.multidict(
    {
        "gen1": [5.0, 0.5, 1.0, 2, 1],
        "gen2": [5.0, 0.5, 0.5, 2, 1],
        "gen3": [5.0, 3.0, 2.0, 2, 1],
    }
)

# operating limits
thermal_units_limits, pmin, pmax = gp.multidict(
    {"gen1": [1.5, 5.0], "gen2": [2.5, 10.0], "gen3": [1.0, 3.0]}
)

# initial commitment status
thermal_units_dyn_data, init_status = gp.multidict(
    {"gen1": [0], "gen2": [0], "gen3": [0]}
)

required_power = [l - s for l, s in zip(load_forecast, solar_forecast)]

def show_results():
    obj_val_s = model.ObjVal
    print(f" OverAll Cost = {round(obj_val_s, 2)}")
    print()
    print("%5s" % "time", end=" ")
    for t in range(nTimeIntervals):
        print("%4s" % t, end=" ")
    print()
    for g in thermal_units:
        print("%5s" % g, end=" ")
        for t in range(nTimeIntervals):
            print("%4.1f" % thermal_units_out_power[g, t].X, end=" ")
        print()
    print("%5s" % "Solar", end=" ")
    for t in range(nTimeIntervals):
        print("%4.1f" % solar_forecast[t], end=" ")
    print()
    print("%5s" % "Load", end=" ")
    for t in range(nTimeIntervals):
        print("%4.1f" % load_forecast[t], end=" ")
    print()

with gp.Env() as env, gp.Model(env=env) as model:
    # variables (indexed, not MVar)
    thermal_units_out_power = model.addVars(
        thermal_units, range(nTimeIntervals), lb=0.0, name="thermal_units_out_power"
    )
    thermal_units_startup_status = model.addVars(
        thermal_units, range(nTimeIntervals), vtype=GRB.BINARY, name="thermal_unit_startup_status"
    )
    thermal_units_shutdown_status = model.addVars(
        thermal_units, range(nTimeIntervals), vtype=GRB.BINARY, name="thermal_unit_shutdown_status"
    )
    thermal_units_comm_status = model.addVars(
        thermal_units, range(nTimeIntervals), vtype=GRB.BINARY, name="thermal_unit_comm_status"
    )

    # objective: sum_{i,t} ( c_i p^2 + b_i p + a_i u + delta_i v + zeta_i w )
    obj_fun_expr = gp.QuadExpr(0.0)
    for t in range(nTimeIntervals):
        for g in thermal_units:
            p = thermal_units_out_power[g, t]
            u = thermal_units_comm_status[g, t]
            v = thermal_units_startup_status[g, t]
            w = thermal_units_shutdown_status[g, t]
            obj_fun_expr += c[g] * (p * p) + b[g] * p + a[g] * u + sup_cost[g] * v + sdn_cost[g] * w
    model.setObjective(obj_fun_expr, GRB.MINIMIZE)

    # power balance: sum_i p_{i,t} == L_t - S_t
    for t in range(nTimeIntervals):
        model.addConstr(
            gp.quicksum(thermal_units_out_power[g, t] for g in thermal_units) == required_power[t],
            name="power_balance_" + str(t),
        )

    # logical constraints
    for t in range(nTimeIntervals):
        for g in thermal_units:
            if t == 0:
                model.addConstr(
                    thermal_units_comm_status[g, 0] - init_status[g]
                    == thermal_units_startup_status[g, 0] - thermal_units_shutdown_status[g, 0],
                    name="logical1_" + g + "_" + str(t),
                )
            else:
                model.addConstr(
                    thermal_units_comm_status[g, t] - thermal_units_comm_status[g, t - 1]
                    == thermal_units_startup_status[g, t] - thermal_units_shutdown_status[g, t],
                    name="logical1_" + g + "_" + str(t),
                )
            model.addConstr(
                thermal_units_startup_status[g, t] + thermal_units_shutdown_status[g, t] <= 1,
                name="logical2_" + g + "_" + str(t),
            )

    # physical limits via indicator constraints
    for t in range(nTimeIntervals):
        for g in thermal_units:
            # if online then p >= pmin
            model.addGenConstrIndicator(
                thermal_units_comm_status[g, t], True,
                thermal_units_out_power[g, t] >= pmin[g],
                name=f"pmin_{g}_{t}"
            )
            # if online then p <= pmax
            model.addGenConstrIndicator(
                thermal_units_comm_status[g, t], True,
                thermal_units_out_power[g, t] <= pmax[g],
                name=f"pmax_{g}_{t}"
            )
            # if offline then p == 0
            model.addGenConstrIndicator(
                thermal_units_comm_status[g, t], False,
                thermal_units_out_power[g, t] == 0.0,
                name=f"poff_{g}_{t}"
            )

    model.optimize()
    if model.SolCount:
        show_results()
