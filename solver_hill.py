# solver_hill.py
"""
Hill Climbing VRP
"""

from copy import deepcopy
import random

PARAMS = {
    "max_iterations": 500,
    "max_no_improve": 100,
    "neighborhood_type": "swap",   # "swap" ou "relocate"
    "seed": 42,
    "verbose": False,
}

def parse_instance(instance):
    n = instance["dimension"]
    depot = instance["depot"][0] - 1
    dist = instance["edge_weight"]
    capacity = instance["capacity"]
    demands = instance["demand"]
    return n, depot, dist, capacity, demands

from solver import solve_vrp as base_solver

def routes_cost(routes, dist):
    return sum(dist[r[i]][r[i+1]] for r in routes for i in range(len(r)-1))

def solve_vrp(instance: dict) -> dict:
    random.seed(PARAMS["seed"])
    n, depot, dist, cap, demands = parse_instance(instance)

    base = base_solver(instance)
    routes = base["routes"]
    best_cost = base["cost"]

    no_improve = 0
    it = 0
    while it < PARAMS["max_iterations"] and no_improve < PARAMS["max_no_improve"]:
        improved = False
        current_best_routes = routes
        current_best_cost = best_cost

        for r_idx, r in enumerate(routes):
            for i in range(1, len(r)-1):
                for j in range(i+1, len(r)-1):
                    new_routes = deepcopy(routes)
                    new_routes[r_idx][i], new_routes[r_idx][j] = new_routes[r_idx][j], new_routes[r_idx][i]
                    new_cost = routes_cost(new_routes, dist)
                    if new_cost < current_best_cost:
                        current_best_cost = new_cost
                        current_best_routes = new_routes
                        improved = True

        routes = current_best_routes
        best_cost = current_best_cost

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        it += 1

    return {"routes": routes, "cost": best_cost}
