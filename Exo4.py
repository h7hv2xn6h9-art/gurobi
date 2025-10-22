import json
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
 
# ----------------------------
# 1. Charger les données
# ----------------------------
with open("data/data/portfolio-example.json", "r") as f:
    data = json.load(f)
 
n = data["num_assets"]
sigma = np.array(data["covariance"])
mu = np.array(data["expected_return"])
mu_0 = data["target_return"]
k = data["portfolio_max_size"]
 
# ----------------------------
# 2. Créer le modèle
# ----------------------------
model = gp.Model("portfolio")
 
# ----------------------------
# 3. Ajouter les variables
# ----------------------------
x = model.addVars(n, lb=0.0, ub=1.0, name="x")      # fractions d'investissement
y = model.addVars(n, vtype=GRB.BINARY, name="y")    # sélection binaire des actifs
 
# ----------------------------
# 4. Définir l'objectif (minimiser le risque)
# ----------------------------
quad = gp.QuadExpr()
for i in range(n):
    for j in range(n):
        quad.add(sigma[i, j] * x[i] * x[j])
model.setObjective(quad, GRB.MINIMIZE)
 
# ----------------------------
# 5. Ajouter les contraintes
# ----------------------------
# 5.1 Rendement minimal
model.addConstr(gp.quicksum(mu[i] * x[i] for i in range(n)) >= mu_0, name="return")
 
# 5.2 Budget total = 1
model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1.0, name="budget")
 
# 5.3 Nombre maximal d'actifs
model.addConstr(gp.quicksum(y[i] for i in range(n)) <= k, name="cardinality")
 
# 5.4 Lien xi <= yi pour chaque actif
for i in range(n):
    model.addConstr(x[i] <= y[i], name=f"link_{i}")
 
# ----------------------------
# 6. Optimiser le modèle
# ----------------------------
model.optimize()
 
# ----------------------------
# 7. Extraire et afficher les résultats
# ----------------------------
if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
    portfolio = [x[i].X for i in range(n)]
    risk = model.ObjVal
    expected_return = float(np.dot(mu, portfolio))
 
    df = pd.DataFrame(
        data=portfolio + [risk, expected_return],
        index=[f"asset_{i}" for i in range(n)] + ["risk", "return"],
        columns=["Portfolio"],
    )
    print(df)
else:
    print("Le modèle n'a pas été résolu en optimum. Statut Gurobi:", model.Status)
    if model.Status == GRB.INFEASIBLE:
        print("Le modèle est infaisable — vérifiez le rendement cible ou la taille maximale du portefeuille.")
 
 