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
 
# global number of time intervals
nTimeIntervals = len(load_forecast)
 
# thermal units
thermal_units = ["gen1", "gen2", "gen3"]
 
# thermal units' costs  (a + b*p + c*p^2), (startup and shutdown costs)
thermal_units_cost, a, b, c, sup_cost, sdn_cost = gp.multidict(
    {
        "gen1": [5.0, 0.5, 1.0, 2, 1],
        "gen2": [5.0, 0.5, 0.5, 2, 1],
        "gen3": [5.0, 3.0, 2.0, 2, 1],
    }
)
 
# thernal units operating limits
thermal_units_limits, pmin, pmax = gp.multidict(
    {"gen1": [1.5, 5.0], "gen2": [2.5, 10.0], "gen3": [1.0, 3.0]}
)
 
# thermal units dynamic data (initial commitment status)
thermal_units_dyn_data, init_status = gp.multidict(
    {"gen1": [0], "gen2": [0], "gen3": [0]}
)
 
 
def show_results():
    obj_val_s = model.ObjVal
    print(f" OverAll Cost = {round(obj_val_s, 2)}	")
    print("\n")
    print("%5s" % "time", end=" ")
    for t in range(nTimeIntervals):
        print("%4s" % t, end=" ")
    print("\n")
 
    for g in thermal_units:
        print("%5s" % g, end=" ")
        for t in range(nTimeIntervals):
            print("%4.1f" % thermal_units_out_power[g, t].X, end=" ")
        print("\n")
 
    print("%5s" % "Solar", end=" ")
    for t in range(nTimeIntervals):
        print("%4.1f" % solar_forecast[t], end=" ")
    print("\n")
 
    print("%5s" % "Load", end=" ")
    for t in range(nTimeIntervals):
        print("%4.1f" % load_forecast[t], end=" ")
    print("\n")
 
 
with gp.Env() as env, gp.Model(env=env) as model:
 
    # add variables for thermal units (power and statuses for commitment, startup and shutdown)
    thermal_units_out_power = model.addVars(
        thermal_units, range(nTimeIntervals), name="thermal_units_out_power"
    )
    thermal_units_startup_status = model.addVars(
        thermal_units,
        range(nTimeIntervals),
        vtype=GRB.BINARY,
        name="thermal_unit_startup_status",
    )
    thermal_units_shutdown_status = model.addVars(
        thermal_units,
        range(nTimeIntervals),
        vtype=GRB.BINARY,
        name="thermal_unit_shutdown_status",
    )
    thermal_units_comm_status = model.addVars(
        thermal_units, range(nTimeIntervals), vtype=GRB.BINARY, name="thermal_unit_comm_status"
    )
 
    # define objective function as an empty quadratic construct and add terms
    obj_fun_expr = gp.QuadExpr(0)
    for t in range(nTimeIntervals):
        for g in thermal_units:
            obj_fun_expr.add(
                a[g] * thermal_units_comm_status[g, t]
                + b[g] * thermal_units_out_power[g, t]
                + c[g] * thermal_units_out_power[g, t] * thermal_units_out_power[g, t]
                + sup_cost[g] * thermal_units_startup_status[g, t]
                + sdn_cost[g] * thermal_units_shutdown_status[g, t]
            )
    model.setObjective(obj_fun_expr)
 
    # Power balance equations
    for t in range(nTimeIntervals):
        model.addConstr(
            gp.quicksum(thermal_units_out_power[g, t] for g in thermal_units)
            == load_forecast[t] - solar_forecast[t],
            name="power_balance_" + str(t),
        )
 
    # Thermal units logical constraints
    for t in range(nTimeIntervals):
        for g in thermal_units:
            if t == 0:
                model.addConstr(
                    thermal_units_comm_status[g, t] - init_status[g]
                    == thermal_units_startup_status[g, t]
                    - thermal_units_shutdown_status[g, t],
                    name="logical1_" + g + "_" + str(t),
                )
            else:
                model.addConstr(
                    thermal_units_comm_status[g, t]
                    - thermal_units_comm_status[g, t - 1]
                    == thermal_units_startup_status[g, t]
                    - thermal_units_shutdown_status[g, t],
                    name="logical1_" + g + "_" + str(t),
                )
 
            model.addConstr(
                thermal_units_startup_status[g, t] + thermal_units_shutdown_status[g, t]
                <= 1,
                name="logical2_" + g + "_" + str(t),
            )
 
    # Thermal units physical constraints, using indicator constraints
    for t in range(nTimeIntervals):
        for g in thermal_units:
            model.addConstr(
                (thermal_units_comm_status[g, t] == 1) >>
                (thermal_units_out_power[g, t] >= pmin[g])
            )
            model.addConstr(
                (thermal_units_comm_status[g, t] == 1) >>
                (thermal_units_out_power[g, t] <= pmax[g])
            )
            model.addConstr(
                (thermal_units_comm_status[g, t] == 0) >>
                (thermal_units_out_power[g, t] == 0.0)
            )
 
    model.optimize()
    show_results()