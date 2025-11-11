# solver_tabu.py
"""
Recherche tabou pour VRP
"""
import random
from copy import deepcopy
from collections import deque
from solver import solve_vrp as base_solver

PARAMS = {
    "max_iterations": 300,
    "tabu_tenure": 50,
    "neighborhood_type": "relocate",
    "aspiration": True,
    "seed": 42,
    "verbose": False,
}

def total_cost(routes, dist):
    return sum(dist[r[i]][r[i+1]] for r in routes for i in range(len(r)-1))

def route_load(route, demands, depot):
    return sum(demands[i] for i in route if i != depot)

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

    tabu = deque(maxlen=PARAMS["tabu_tenure"])

    for _ in range(PARAMS["max_iterations"]):
        best_move = None
        best_candidate = None
        best_candidate_cost = float("inf")

        for r_idx, r in enumerate(current):
            for pos in range(1, len(r)-1):
                client = r[pos]
                for r2_idx in range(len(current)):
                    for pos2 in range(1, len(current[r2_idx])):
                        move = (r_idx, pos, r2_idx, pos2)
                        if move in tabu:
                            continue
                        cand = deepcopy(current)
                        c = cand[r_idx].pop(pos)
                        cand[r2_idx].insert(pos2, c)

                        # capacité
                        if route_load(cand[r2_idx], demands, depot) > capacity:
                            continue

                        c_cost = total_cost(cand, dist)
                        if c_cost < best_candidate_cost:
                            best_candidate_cost = c_cost
                            best_candidate = cand
                            best_move = move

        if best_candidate is None:
            break

        current = best_candidate
        current_cost = best_candidate_cost
        tabu.append(best_move)

        # aspiration
        if current_cost < best_cost:
            best = deepcopy(current)
            best_cost = current_cost

    return {"routes": best, "cost": best_cost}
