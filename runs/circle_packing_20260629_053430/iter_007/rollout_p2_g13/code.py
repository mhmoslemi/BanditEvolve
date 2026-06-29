import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a hexagonal grid pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint functions for boundary and overlap
    def boundary_constraints(v):
        return [
            v[3*i] - v[3*i+2] for i in range(n)
        ] + [
            1.0 - v[3*i] - v[3*i+2] for i in range(n)
        ] + [
            v[3*i+1] - v[3*i+2] for i in range(n)
        ] + [
            1.0 - v[3*i+1] - v[3*i+2] for i in range(n)
        ]

    def overlap_constraints(v):
        dist_sq = (v[3*np.arange(n)[:, np.newaxis]] - v[3*np.arange(n)])**2 + \
                  (v[3*np.arange(n)+1][:, np.newaxis] - v[3*np.arange(n)+1])**2
        min_dist_sq = (v[3*np.arange(n)+2][:, np.newaxis] + v[3*np.arange(n)+2])**2
        return dist_sq - min_dist_sq

    # Define constraints
    cons = []
    for i in range(n):
        for expr in boundary_constraints(v0):
            cons.append({"type": "ineq", "fun": lambda v, i=i: expr})
    for expr in overlap_constraints(v0):
        cons.append({"type": "ineq", "fun": lambda v, expr=expr: expr})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-8})
    v = res.x if res.success else v0

    # Phase 2: Apply geometric transformation and perturbation
    scale_factor = 1.1
    v_transformed = v.copy()
    v_transformed[0::3] *= scale_factor
    v_transformed[1::3] *= scale_factor
    v_transformed[1::3] += 0.1  # slight vertical shift to break symmetry
    v_transformed[2::3] *= scale_factor

    # Rebuild bounds and constraints for transformed configuration
    bounds_transformed = []
    for _ in range(n):
        bounds_transformed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    cons_transformed = []
    for i in range(n):
        for expr in boundary_constraints(v_transformed):
            cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: expr})
    for expr in overlap_constraints(v_transformed):
        cons_transformed.append({"type": "ineq", "fun": lambda v, expr=expr: expr})

    res_transformed = minimize(neg_sum_radii, v_transformed, method="SLSQP", bounds=bounds_transformed,
                              constraints=cons_transformed, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_transformed.x if res_transformed.success else v

    # Phase 3: Local refinement with controlled perturbation
    perturbation = 0.03
    v_perturbed = v.copy()
    np.random.seed(42)
    for i in range(n):
        v_perturbed[3*i] += np.random.uniform(-perturbation, perturbation)
        v_perturbed[3*i+1] += np.random.uniform(-perturbation, perturbation)
        v_perturbed[3*i+2] += np.random.uniform(-perturbation, perturbation)

    # Rebuild bounds and constraints for perturbed configuration
    bounds_perturbed = []
    for _ in range(n):
        bounds_perturbed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    cons_perturbed = []
    for i in range(n):
        for expr in boundary_constraints(v_perturbed):
            cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: expr})
    for expr in overlap_constraints(v_perturbed):
        cons_perturbed.append({"type": "ineq", "fun": lambda v, expr=expr: expr})

    res_perturbed = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds_perturbed,
                             constraints=cons_perturbed, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_perturbed.x if res_perturbed.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())