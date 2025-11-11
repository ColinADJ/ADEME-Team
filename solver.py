# solver.py
"""
Solveur glouton de base pour VRPLIB.
"""

PARAMS = {
    "nearest_neighbor": True,
    "seed": 42,
    "verbose": False,
}

import random

def parse_instance(instance: dict):
    n = instance["dimension"]
    depot = instance["depot"][0] - 1  # 0-based
    dist = instance["edge_weight"]
    capacity = instance["capacity"]
    demands = instance["demand"]
    return n, depot, dist, capacity, demands


def solve_vrp(instance: dict) -> dict:
    random.seed(PARAMS["seed"])
    n, depot, dist, capacity, demands = parse_instance(instance)

    clients = [i for i in range(n) if i != depot]
    routes = []
    total_cost = 0.0

    while clients:
        route = [depot]
        load = 0
        current = depot

        while True:
            feasible = [c for c in clients if load + demands[c] <= capacity]
            if not feasible:
                break

            if PARAMS["nearest_neighbor"]:
                next_client = min(feasible, key=lambda c: dist[current][c])
            else:
                next_client = random.choice(feasible)

            route.append(next_client)
            total_cost += dist[current][next_client]
            load += demands[next_client]
            current = next_client
            clients.remove(next_client)

        route.append(depot)
        total_cost += dist[current][depot]
        routes.append(route)

    return {"routes": routes, "cost": total_cost}
