import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
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

    # Define constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints
    i_indices = np.arange(n)
    j_indices = np.triu_indices(n, 1)
    dx = v0[0::3][i_indices] - v0[0::3][j_indices]
    dy = v0[1::3][i_indices] - v0[1::3][j_indices]
    dist_sq = dx**2 + dy**2
    min_dist_sq = (v0[2::3][i_indices] + v0[2::3][j_indices])**2
    overlap_constraints = dist_sq - min_dist_sq
    for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
        cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: v[3*i] - v[3*j] - v[3*i+2] - v[3*j+2]})

    # Phase 1: Initial optimization with initial layout
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

    bounds_transformed = []
    for _ in range(n):
        bounds_transformed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_transformed = []
    for i in range(n):
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints for transformed configuration
    i_indices = np.arange(n)
    j_indices = np.triu_indices(n, 1)
    dx = v_transformed[0::3][i_indices] - v_transformed[0::3][j_indices]
    dy = v_transformed[1::3][i_indices] - v_transformed[1::3][j_indices]
    dist_sq = dx**2 + dy**2
    min_dist_sq = (v_transformed[2::3][i_indices] + v_transformed[2::3][j_indices])**2
    overlap_constraints = dist_sq - min_dist_sq
    for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i, j=j: v[3*i] - v[3*j] - v[3*i+2] - v[3*j+2]})

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

    bounds_perturbed = []
    for _ in range(n):
        bounds_perturbed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_perturbed = []
    for i in range(n):
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints for perturbed configuration
    i_indices = np.arange(n)
    j_indices = np.triu_indices(n, 1)
    dx = v_perturbed[0::3][i_indices] - v_perturbed[0::3][j_indices]
    dy = v_perturbed[1::3][i_indices] - v_perturbed[1::3][j_indices]
    dist_sq = dx**2 + dy**2
    min_dist_sq = (v_perturbed[2::3][i_indices] + v_perturbed[2::3][j_indices])**2
    overlap_constraints = dist_sq - min_dist_sq
    for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i, j=j: v[3*i] - v[3*j] - v[3*i+2] - v[3*j+2]})

    res_perturbed = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds_perturbed,
                             constraints=cons_perturbed, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_perturbed.x if res_perturbed.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())