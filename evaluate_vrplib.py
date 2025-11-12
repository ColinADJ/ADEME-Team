# evaluate_vrplib.py
"""
Évaluation comparative de plusieurs algorithmes VRP :
Glouton, Hill Climbing, LNS, ACO, Recuit, Tabou, ALNS
avec mesure du gap et du temps de calcul.
"""

import csv
import json
import os
import time
import vrplib

# === IMPORTS DES SOLVEURS ===
from solver import solve_vrp as solve_glouton, PARAMS as PARAMS_GLOUTON
from solver_hill import solve_vrp as solve_hill, PARAMS as PARAMS_HILL
from solver_lns import solve_vrp as solve_lns, PARAMS as PARAMS_LNS
from solver_aco import solve_vrp as solve_aco, PARAMS as PARAMS_ACO
from solver_sa import solve_vrp as solve_sa, PARAMS as PARAMS_SA
from solver_tabu import solve_vrp as solve_tabu, PARAMS as PARAMS_TABU
from alns_solver_vrplib import solve_vrp as solve_alns, PARAMS as PARAMS_ALNS


# ================================
# 1. Fonctions utilitaires
# ================================

def compute_gap(my_cost, opt_cost):
    """Calcule le gap relatif à la solution optimale."""
    return 100.0 * (my_cost - opt_cost) / opt_cost


def evaluate_instance(instance_file, solution_file):
    """Teste tous les solveurs sur une instance donnée."""
    instance = vrplib.read_instance(instance_file)
    optimal = vrplib.read_solution(solution_file)
    opt_cost = optimal["cost"]

    solvers = [
        ##("Glouton", solve_glouton, PARAMS_GLOUTON),
        ##("Hill climbing", solve_hill, PARAMS_HILL),
        ##("LNS", solve_lns, PARAMS_LNS),
        ##("Colonies de fourmis", solve_aco, PARAMS_ACO),
        #"("Recuit simulé", solve_sa, PARAMS_SA),
        ##("Recherche tabou", solve_tabu, PARAMS_TABU),
        ("ALNS", solve_alns, PARAMS_ALNS),
    ]

    results = []
    print(f"\n=== Instance : {instance_file} ===")
    print(f"{'Algorithme':<22} {'Coût opti':<10} {'Coût obtenu':<14} {'Gap (%)':<8} {'Temps (s)':<10}")
    print("-" * 70)

    for name, solver_fun, params in solvers:
        if name == "ALNS":
            # 🔁 20 runs indépendants avec graines différentes
            for seed in range(20):
                params["seed"] = seed
                start = time.time()
                sol = solver_fun(instance)
                end = time.time()

                my_cost = sol["cost"]
                elapsed = end - start
                gap = compute_gap(my_cost, opt_cost)

                print(f"{name+'_'+str(seed):<22} {opt_cost:<10} {my_cost:<14.2f} {gap:<8.2f} {elapsed:<10.3f}")

                results.append({
                    "instance": instance_file,
                    "algo": name,
                    "seed": seed,
                    "cost_opt": opt_cost,
                    "cost": my_cost,
                    "gap": gap,
                    "time_s": elapsed,
                    "params": json.dumps(params)
                })

        else:
            # autres solveurs : 1 seul run
            start = time.time()
            sol = solver_fun(instance)
            end = time.time()

            my_cost = sol["cost"]
            elapsed = end - start
            gap = compute_gap(my_cost, opt_cost)

            print(f"{name:<22} {opt_cost:<10} {my_cost:<14.2f} {gap:<8.2f} {elapsed:<10.3f}")

            results.append({
                "instance": instance_file,
                "algo": name,
                "seed": None,
                "cost_opt": opt_cost,
                "cost": my_cost,
                "gap": gap,
                "time_s": elapsed,
                "params": json.dumps(params)
            })
    return results


# ================================
# 2. Boucle principale
# ================================

def main():
    tests = [
        ("A-n32-k5.vrp", "A-n32-k5.sol"),
        ("X-n101-k25.vrp", "X-n101-k25.sol"),
        ("ORTEC-n242-k12.vrp", "ORTEC-n242-k12.sol"),
    ]

    all_results = []

    for inst, sol in tests:
        all_results.extend(evaluate_instance(inst, sol))

    # === Écriture CSV ===
    out_file = "results.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance",
                "algo",
                "cost_opt",
                "cost",
                "gap",
                "time_s",
                "params"
            ]
        )
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print("\n✅ Résultats enregistrés dans :", os.path.abspath(out_file))


# ================================
# 3. Lancement
# ================================
if __name__ == "__main__":
    main()
