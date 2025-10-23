import math
import gurobipy as gp
from gurobipy import GRB, nlfunc as nl

# Parameters
L1, L2 = 1.0, 0.8             # Lengths of the links
x_star, y_star = 1.20, 0.60   # Point to reach
xo, yo, r = 0.50, 0.00, 0.20  # Disk to avoid

# Joint limits
theta1_min, theta1_max = -math.pi, math.pi
theta2_min, theta2_max = -0.75*math.pi, 0.75*math.pi

# Build model
m = gp.Model("robot_arm_nlfunc")

# Gurobi doit autoriser le non convexe (contrainte >= pour le disque)
m.Params.NonConvex = 2

# Decision variables (angles)
theta1 = m.addVar(lb=theta1_min, ub=theta1_max, name="theta1")
theta2 = m.addVar(lb=theta2_min, ub=theta2_max, name="theta2")

# Variables pour l'EEF et le milieu du 1er lien
x  = m.addVar(name="x")
y  = m.addVar(name="y")
xm = m.addVar(name="xm")   # midpoint x of link 1
ym = m.addVar(name="ym")   # midpoint y of link 1

# Cinématique directe (EEF)
m.addConstr(x == L1*nl.cos(theta1)            + L2*nl.cos(theta1 + theta2), name="fk_x")
m.addConstr(y == L1*nl.sin(theta1)            + L2*nl.sin(theta1 + theta2), name="fk_y")

# Milieu du premier lien
m.addConstr(xm == 0.5*L1*nl.cos(theta1), name="mid_x")
m.addConstr(ym == 0.5*L1*nl.sin(theta1), name="mid_y")

# Atteindre la cible (égalité)
m.addConstr(x == x_star, name="reach_x")
m.addConstr(y == y_star, name="reach_y")

# Avoid circular obstacle: (xm - xo)^2 + (ym - yo)^2 >= r^2  (non convexe)
m.addQConstr((xm - xo)*(xm - xo) + (ym - yo)*(ym - yo) >= r*r, name="avoid_disk")

# Objectif: faisabilité pure (0)
m.setObjective(0.0, GRB.MINIMIZE)

m.optimize()

sol = None
if m.Status == GRB.OPTIMAL:
    sol = {
        "theta1": theta1.X, "theta2": theta2.X,
        "x": x.X, "y": y.X, "xm": xm.X, "ym": ym.X,
        "obj": m.ObjVal,
    }
    print("Optimal objective:", m.ObjVal)
    print(sol)
else:
    print("Optimization status:", m.Status)
