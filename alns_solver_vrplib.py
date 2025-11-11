# alns_solver_vrplib.py
"""
ALNS adapté aux instances VRPLIB
- coût = somme des distances (comme dans les .sol VRPLIB)
- adaptation du nb d'itérations à la taille
- destruction aléatoire
- réparation cheap + réparation rapide
- petite amélioration locale 2-opt
- post-traitement pour respecter le nombre de véhicules de l'instance

Point d'entrée : solve_vrp(instance_vrplib)
"""

import random
from copy import deepcopy

# ===========================
# Paramètres globaux
# ===========================
PARAMS = {
    # nombre d'itérations selon la taille
    "iters_small": 3500,   # ~A-n32-k5
    "iters_medium": 5000,  # ~X-n101-k25
    "iters_large": 7000,   # ORTEC 200+
    # paramètres ALNS
    "destroy_ratio": 0.30,
    "start_temp": 500.0,
    "cooling": 0.995,
    # générique
    "seed": 42,
    "verbose": False,
}

# ===========================
# Utilitaires VRP
# ===========================
def route_load(route, inst):
    demands = inst["demands"]
    return sum(demands[i] for i in route if i != 0)


def cost_vrplib(solution, dist):
    total = 0.0
    for r in solution:
        for i in range(len(r) - 1):
            total += dist[r[i]][r[i + 1]]
    return total


# ===========================
# Construction initiale
# ===========================
def initial_solution_vrp(inst):
    """
    Construction gloutonne simple : on met les clients dans la première
    route faisable, sinon on crée une nouvelle route.
    """
    n_clients = inst["n_clients"]
    capacity = inst["capacity"]
    depot = 0

    clients = list(range(1, n_clients + 1))
    random.shuffle(clients)

    solution = []
    for c in clients:
        placed = False
        for idx, r in enumerate(solution):
            cand = r[:-1] + [c, depot]
            if route_load(cand, inst) <= capacity:
                solution[idx] = cand
                placed = True
                break
        if not placed:
            solution.append([depot, c, depot])
    return solution


# ===========================
# Destroy / Repair
# ===========================
def destroy_random(solution, ratio=0.1):
    """retire un pourcentage de clients au hasard"""
    all_clients = []
    for r in solution:
        for x in r:
            if x != 0:
                all_clients.append(x)

    n_remove = max(1, int(len(all_clients) * ratio))
    removed = set(random.sample(all_clients, n_remove))

    new_sol = []
    for r in solution:
        nr = [x for x in r if x not in removed]
        if nr[0] != 0:
            nr.insert(0, 0)
        if nr[-1] != 0:
            nr.append(0)
        new_sol.append(nr)

    return new_sol, list(removed)


def repair_cheapest_vrp(solution, removed, inst):
    """réparation complète (on teste toutes les positions)"""
    dist = inst["traffic_matrices"][0]
    cap = inst["capacity"]
    depot = 0

    for c in removed:
        best_cost = None
        best_r = None
        best_pos = None

        for r_idx, r in enumerate(solution):
            for pos in range(1, len(r)):
                cand = r[:pos] + [c] + r[pos:]
                if route_load(cand, inst) <= cap:
                    # coût de la route candidate
                    local = 0.0
                    for i in range(len(cand) - 1):
                        local += dist[cand[i]][cand[i + 1]]
                    if best_cost is None or local < best_cost:
                        best_cost = local
                        best_r = r_idx
                        best_pos = pos

        if best_r is None:
            solution.append([depot, c, depot])
        else:
            r = solution[best_r]
            solution[best_r] = r[:best_pos] + [c] + r[best_pos:]

    return solution


def repair_fast(solution, removed, inst, max_pos_per_route=5):
    """réparation plus rapide : teste seulement quelques positions"""
    dist = inst["traffic_matrices"][0]
    cap = inst["capacity"]
    depot = 0

    for c in removed:
        best_cost = None
        best_r = None
        best_pos = None

        for r_idx, r in enumerate(solution):
            limit = min(len(r), max_pos_per_route)
            for pos in range(1, limit):
                cand = r[:pos] + [c] + r[pos:]
                if route_load(cand, inst) <= cap:
                    local = 0.0
                    for i in range(len(cand) - 1):
                        local += dist[cand[i]][cand[i + 1]]
                    if best_cost is None or local < best_cost:
                        best_cost = local
                        best_r = r_idx
                        best_pos = pos

        if best_r is None:
            solution.append([depot, c, depot])
        else:
            r = solution[best_r]
            solution[best_r] = r[:best_pos] + [c] + r[best_pos:]

    return solution


# ===========================
# 2-opt local
# ===========================
def two_opt_route(route, dist):
    """2-opt basique sur une route"""
    best_route = route
    best_cost = sum(dist[route[i]][route[i + 1]] for i in range(len(route) - 1))
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                new_cost = sum(dist[new_route[k]][new_route[k + 1]] for k in range(len(new_route) - 1))
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_route = new_route
                    improved = True
                    break
            if improved:
                break
    return best_route


def local_improvement(solution, inst):
    dist = inst["traffic_matrices"][0]
    improved = []
    for r in solution:
        if len(r) > 3:
            improved.append(two_opt_route(r, dist))
        else:
            improved.append(r)
    return improved


# ===========================
# Fusion de routes pour respecter k véhicules
# ===========================
def merge_routes_until_k(routes, inst, k, dist):
    """
    Si on a plus de routes que k, on essaie de fusionner les plus petites
    tant que c'est possible sans dépasser la capacité.
    """
    cap = inst["capacity"]
    demands = inst["demands"]

    def load(r):
        return sum(demands[i] for i in r if i != 0)

    while len(routes) > k:
        # trier par route la plus petite
        routes = sorted(routes, key=load)
        small = routes[0]
        load_small = load(small)

        best_idx = None
        best_inc = None
        best_new_route = None

        # essayer de la coller dans une autre route
        for r_idx in range(1, len(routes)):
            r = routes[r_idx]
            if load(r) + load_small > cap:
                continue
            # on colle le contenu de small (sans 0) avant le dernier 0
            content = [x for x in small if x != 0]
            cand = r[:-1] + content + [0]

            old_cost = sum(dist[r[i]][r[i + 1]] for i in range(len(r) - 1))
            new_cost = sum(dist[cand[i]][cand[i + 1]] for i in range(len(cand) - 1))
            inc = new_cost - old_cost

            if best_inc is None or inc < best_inc:
                best_inc = inc
                best_idx = r_idx
                best_new_route = cand

        if best_idx is not None:
            routes[best_idx] = best_new_route
            routes.pop(0)
        else:
            # impossible de fusionner plus sans casser la capacité
            break

    return routes


# ===========================
# ALNS principal (version VRPLIB)
# ===========================
def alns_vrplib(inst,
                iters=1000,
                base_destroy=0.1,
                start_temp=500.0,
                cooling=0.995):
    dist = inst["traffic_matrices"][0]

    current = initial_solution_vrp(inst)
    current_cost = cost_vrplib(current, dist)
    best = deepcopy(current)
    best_cost = current_cost

    T = start_temp

    for _ in range(iters):
        # 1) destruction avec petite variation
        dr = base_destroy * random.uniform(0.8, 1.25)
        partial, removed = destroy_random(current, ratio=dr)

        # 2) réparation : parfois complète, parfois rapide
        if random.random() < 0.3:
            candidate = repair_cheapest_vrp(partial, removed, inst)
        else:
            candidate = repair_fast(partial, removed, inst)

        # 3) amélioration locale
        candidate = local_improvement(candidate, inst)

        cand_cost = cost_vrplib(candidate, dist)

        # 4) acceptation
        if cand_cost < current_cost:
            current = candidate
            current_cost = cand_cost
            if cand_cost < best_cost:
                best = deepcopy(candidate)
                best_cost = cand_cost
        else:
            delta = cand_cost - current_cost
            prob = (pow(2.71828, -delta / T)) if T > 1e-9 else 0
            if random.random() < prob:
                current = candidate
                current_cost = cand_cost

        T *= cooling

    return best, best_cost


# ===========================
# Point d'entrée pour evaluate_vrplib.py
# ===========================
def solve_vrp(instance_vrplib: dict) -> dict:
    random.seed(PARAMS["seed"])

    n = instance_vrplib["dimension"]
    dist = instance_vrplib["edge_weight"]
    cap = instance_vrplib["capacity"]
    raw_demands = instance_vrplib["demand"]
    k_veh = instance_vrplib.get("vehicles", 999)

    # normalisation des demandes (certaines instances sont 1-based)
    if len(raw_demands) == n:
        demands = list(raw_demands)
    else:
        demands = [raw_demands[i + 1] for i in range(n)]

    inst = {
        "n_nodes": n,
        "n_clients": n - 1,
        "capacity": cap,
        "demands": demands,
        "traffic_matrices": [dist],
    }

    # choisir le bon nombre d'itérations
    if n <= 60:
        iters = PARAMS["iters_small"]
    elif n <= 200:
        iters = PARAMS["iters_medium"]
    else:
        iters = PARAMS["iters_large"]

    routes, cost = alns_vrplib(
        inst,
        iters=iters,
        base_destroy=PARAMS["destroy_ratio"],
        start_temp=PARAMS["start_temp"],
        cooling=PARAMS["cooling"]
    )

    # post-traitement pour respecter le nombre de véhicules
    if k_veh is not None and k_veh < len(routes):
        routes = merge_routes_until_k(routes, inst, k_veh, dist)
        cost = cost_vrplib(routes, dist)

    return {
        "routes": routes,
        "cost": cost
    }
