# solver_sa.py
"""
Recuit simulé pour VRP
"""
import random
import math
from copy import deepcopy
from solver import solve_vrp as base_solver

PARAMS = {
    "T0": 1000.0,
    "Tmin": 0.01,
    "cooling_rate": 0.995,
    "max_iterations": 2000,
    "move_type": "relocate",   # "swap" ou "relocate"
    "seed": 42,
    "verbose": False,
}

def total_cost(routes, dist):
    return sum(dist[r[i]][r[i+1]] for r in routes for i in range(len(r)-1))

def solve_vrp(instance: dict) -> dict:
    random.seed(PARAMS["seed"])
    dist = instance["edge_weight"]
    demands = instance["demand"]
    capacity = instance["capacity"]
    depot = instance["depot"][0] - 1

    base = base_solver(instance)
    current = base["routes"]
    current_cost = base["cost"]
    best = deepcopy(current)
    best_cost = current_cost

    T = PARAMS["T0"]
    it = 0
    while T > PARAMS["Tmin"] and it < PARAMS["max_iterations"]:
        cand = deepcopy(current)

        # choisir un client à déplacer
        positions = []
        for r_idx, r in enumerate(cand):
            for pos in range(1, len(r)-1):
                positions.append((r_idx, pos))
        if not positions:
            break

        r_idx, pos = random.choice(positions)
        client = cand[r_idx].pop(pos)

        dest_idx = random.randrange(len(cand))
        dest_route = cand[dest_idx]
        insert_pos = random.randint(1, len(dest_route)-1)
        dest_route.insert(insert_pos, client)

        # vérifier capacité
        def load(route):
            return sum(demands[i] for i in route if i != depot)
        if load(dest_route) > capacity:
            # on annule ce move
            it += 1
            T *= PARAMS["cooling_rate"]
            continue

        cand_cost = total_cost(cand, dist)

        if cand_cost < current_cost:
            current, current_cost = cand, cand_cost
            if cand_cost < best_cost:
                best, best_cost = cand, cand_cost
        else:
            delta = cand_cost - current_cost
            prob = math.exp(-delta / T)
            if random.random() < prob:
                current, current_cost = cand, cand_cost

        T *= PARAMS["cooling_rate"]
        it += 1

    return {"routes": best, "cost": best_cost}
