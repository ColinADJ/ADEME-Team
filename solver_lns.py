# solver_lns.py
"""
LNS simple pour VRP
"""
import random
from copy import deepcopy
from solver import solve_vrp as base_solver

PARAMS = {
    "max_iterations": 300,
    "destroy_rate": 0.12,        # % clients enlevés
    "repair_method": "greedy",   # "greedy" ou "random"
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

def route_load(route, demands, depot):
    return sum(demands[i] for i in route if i != depot)

def routes_cost(routes, dist):
    return sum(dist[r[i]][r[i+1]] for r in routes for i in range(len(r)-1))

def solve_vrp(instance: dict) -> dict:
    random.seed(PARAMS["seed"])
    n, depot, dist, cap, demands = parse_instance(instance)

    base = base_solver(instance)
    routes = base["routes"]
    best_cost = base["cost"]

    for _ in range(PARAMS["max_iterations"]):
        # collect all clients
        all_clients = []
        for r in routes:
            for x in r:
                if x != depot:
                    all_clients.append(x)
        if not all_clients:
            break

        n_remove = max(1, int(len(all_clients) * PARAMS["destroy_rate"]))
        removed = set(random.sample(all_clients, n_remove))

        # destroy
        partial = []
        for r in routes:
            nr = [x for x in r if x not in removed]
            if nr[0] != depot:
                nr.insert(0, depot)
            if nr[-1] != depot:
                nr.append(depot)
            partial.append(nr)

        # repair
        for c in removed:
            best_r, best_pos, best_inc = None, None, None
            for r_idx, r in enumerate(partial):
                for pos in range(1, len(r)):
                    cand = r[:pos] + [c] + r[pos:]
                    if route_load(cand, demands, depot) <= cap:
                        # coût local
                        local = 0.0
                        for i in range(len(cand)-1):
                            local += dist[cand[i]][cand[i+1]]
                        if best_inc is None or local < best_inc:
                            best_inc = local
                            best_r = r_idx
                            best_pos = pos
            if best_r is None:
                partial.append([depot, c, depot])
            else:
                r = partial[best_r]
                partial[best_r] = r[:best_pos] + [c] + r[best_pos:]

        new_cost = routes_cost(partial, dist)
        if new_cost < best_cost:
            best_cost = new_cost
            routes = partial

    return {"routes": routes, "cost": best_cost}
