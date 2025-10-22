import gurobipy as gp
from gurobipy import GRB
 
import numpy as np
 
 
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
 

thermal_units_limits, pmin, pmax = gp.multidict(
    {"gen1": [1.5, 5.0], "gen2": [2.5, 10.0], "gen3": [1.0, 3.0]}
)
 
# thermal units dynamic data (initial commitment status)
thermal_units_dyn_data, init_status = gp.multidict(
    {"gen1": [0], "gen2": [0], "gen3": [0]}
)
 
 
# We need np.array instances instead of dict instances to use
# the matrix API
def dict_to_array(keys, d):
    return np.array([d[k] for k in keys])
 
a = dict_to_array(thermal_units, a)
b = dict_to_array(thermal_units, b)
c = dict_to_array(thermal_units, c)
sup_cost = dict_to_array(thermal_units, sup_cost)
sdn_cost = dict_to_array(thermal_units, sdn_cost)
pmin = dict_to_array(thermal_units, pmin)
pmax = dict_to_array(thermal_units, pmax)
init_status = dict_to_array(thermal_units, init_status)
 
# We also need to turn to integer indices instead of strings
units_to_index = {u: idx for idx, u in enumerate(thermal_units)}
nb_units = len(thermal_units)
 
 
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
            print("%4.1f" % thermal_units_out_power[units_to_index[g], t].X, end=" ")
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
    thermal_units_out_power = model.addMVar(
        (nb_units, nTimeIntervals), name="thermal_units_out_power"
    )
    thermal_units_startup_status = model.addMVar(
        (nb_units, nTimeIntervals),
        vtype=GRB.BINARY,
        name="thermal_unit_startup_status",
    )
    thermal_units_shutdown_status = model.addMVar(
        (nb_units, nTimeIntervals),
        vtype=GRB.BINARY,
        name="thermal_unit_shutdown_status",
    )
    thermal_units_comm_status = model.addMVar(
        (nb_units, nTimeIntervals), vtype=GRB.BINARY, name="thermal_unit_comm_status"
    )
 
    # define objective function as an empty quadratic construct and add terms
    model.setObjective((a[:, None] * thermal_units_comm_status).sum() +
                       (b[:, None] * thermal_units_out_power).sum() +
                       (c[:, None] * (thermal_units_out_power * thermal_units_out_power)).sum() +
                       (sup_cost[:, None] * thermal_units_startup_status).sum() +
                       (sdn_cost[:, None] * thermal_units_shutdown_status).sum())
 
    # Power balance equations
    required_power = np.array([l - s for l, s in zip(load_forecast, solar_forecast) ])
    model.addConstr(
        thermal_units_out_power.sum(axis=0) == required_power,
        name="power_balance"
    )
 
    # Thermal units logical constraints
 
    # Initial condition (t = 0)
    model.addConstr(
        thermal_units_comm_status[:, 0] - init_status ==
        thermal_units_startup_status[:, 0] - thermal_units_shutdown_status[:, 0],
        name="logical1_initial"
    )
 
    # Remaining time intervals (t > 0)
    model.addConstr(
        thermal_units_comm_status[:, 1:] - thermal_units_comm_status[:, :-1]
        == thermal_units_startup_status[:, 1:] - thermal_units_shutdown_status[:, 1:],
        name="logical1_remaining"
    )
 
    model.addConstr(
        thermal_units_startup_status + thermal_units_shutdown_status <= 1,
        name="logical2"
    )
 
    # Thermal units physical constraints, using indicator constraints
    for g in range(len(thermal_units)):
        model.addGenConstrIndicator(thermal_units_comm_status[g, :], True,
                                    thermal_units_out_power[g, :] >= pmin[g],
                                    name="physical_min")
        model.addGenConstrIndicator(thermal_units_comm_status[g, :], True,
                                    thermal_units_out_power[g, :] <= pmax[g],
                                    name="physical_max")
        model.addGenConstrIndicator(thermal_units_comm_status[g, :], False,
                                    thermal_units_out_power[g, :] == 0.0,
                                    name="physical_off")
 
    model.optimize()
    show_results()