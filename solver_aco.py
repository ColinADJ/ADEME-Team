# solver_aco.py
"""
Colonies de fourmis pour VRP (version simplifiée)
"""
import random
import math

PARAMS = {
    "n_ants": 12,
    "n_iter": 80,
    "alpha": 1.0,
    "beta": 2.0,
    "rho": 0.5,
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

def solve_vrp(instance: dict) -> dict:
    random.seed(PARAMS["seed"])
    n, depot, dist, cap, demands = parse_instance(instance)

    pher = [[1.0 for _ in range(n)] for _ in range(n)]

    def build_solution():
        unserved = [i for i in range(n) if i != depot]
        routes = []
        total = 0.0
        while unserved:
            route = [depot]
            load = 0
            cur = depot
            while True:
                candidates = [c for c in unserved if load + demands[c] <= cap]
                if not candidates:
                    break
                scores = []
                s = 0.0
                for c in candidates:
                    tau = pher[cur][c] ** PARAMS["alpha"]
                    eta = (1.0 / (dist[cur][c] + 1e-6)) ** PARAMS["beta"]
                    val = tau * eta
                    scores.append(val)
                    s += val
                r = random.random() * s
                acc = 0.0
                chosen = candidates[0]
                for c, v in zip(candidates, scores):
                    acc += v
                    if acc >= r:
                        chosen = c
                        break
                route.append(chosen)
                total += dist[cur][chosen]
                load += demands[chosen]
                cur = chosen
                unserved.remove(chosen)
            route.append(depot)
            total += dist[cur][depot]
            routes.append(route)
        return routes, total

    best_routes, best_cost = build_solution()

    for _ in range(PARAMS["n_iter"]):
        sols = []
        for _ in range(PARAMS["n_ants"]):
            rts, c = build_solution()
            sols.append((rts, c))
            if c < best_cost:
                best_cost = c
                best_routes = rts

        # évaporation
        for i in range(n):
            for j in range(n):
                pher[i][j] *= (1 - PARAMS["rho"])

        # dépôt
        for rts, c in sols:
            for r in rts:
                for i in range(len(r)-1):
                    a = r[i]; b = r[i+1]
                    pher[a][b] += 1.0 / (1.0 + c)

    return {"routes": best_routes, "cost": best_cost}
