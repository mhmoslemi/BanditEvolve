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

    # Define constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorize overlap constraints
    i_indices, j_indices = np.triu_indices(n, 1)
    dx = v0[0::3][:, np.newaxis] - v0[0::3][np.newaxis, :]
    dy = v0[1::3][:, np.newaxis] - v0[1::3][np.newaxis, :]
    dist_sq = dx**2 + dy**2
    min_dist_sq = (v0[2::3] + v0[2::3][np.newaxis, :])**2
    overlap_constraints = dist_sq - min_dist_sq
    overlap_constraints = overlap_constraints[i_indices, j_indices]
    cons_overlap = [{"type": "ineq", "fun": lambda v, idx=idx: overlap_constraints[idx]} for idx in range(len(overlap_constraints))]

    cons += cons_overlap

    # Phase 1: Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-8})
    v = res.x if res.success else v0

    # Phase 2: Apply geometric transformation and perturbation
    scale_factor = 1.1
    rotated_v = v.copy()
    rotated_v[0::3] *= scale_factor
    rotated_v[1::3] *= scale_factor
    rotated_v[1::3] += 0.1  # small vertical shift to break symmetry
    rotated_v[2::3] *= scale_factor

    # Rebuild bounds and constraints for transformed configuration
    bounds_transformed = []
    for _ in range(n):
        bounds_transformed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_transformed = []
    for i in range(n):
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorize overlap constraints for transformed configuration
    i_indices, j_indices = np.triu_indices(n, 1)
    dx = rotated_v[0::3][:, np.newaxis] - rotated_v[0::3][np.newaxis, :]
    dy = rotated_v[1::3][:, np.newaxis] - rotated_v[1::3][np.newaxis, :]
    dist_sq = dx**2 + dy**2
    min_dist_sq = (rotated_v[2::3] + rotated_v[2::3][np.newaxis, :])**2
    overlap_constraints_transformed = dist_sq - min_dist_sq
    overlap_constraints_transformed = overlap_constraints_transformed[i_indices, j_indices]
    cons_overlap_transformed = [{"type": "ineq", "fun": lambda v, idx=idx: overlap_constraints_transformed[idx]} for idx in range(len(overlap_constraints_transformed))]

    cons_transformed += cons_overlap_transformed

    res_transformed = minimize(neg_sum_radii, rotated_v, method="SLSQP", bounds=bounds_transformed,
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
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorize overlap constraints for perturbed configuration
    i_indices, j_indices = np.triu_indices(n, 1)
    dx = v_perturbed[0::3][:, np.newaxis] - v_perturbed[0::3][np.newaxis, :]
    dy = v_perturbed[1::3][:, np.newaxis] - v_perturbed[1::3][np.newaxis, :]
    dist_sq = dx**2 + dy**2
    min_dist_sq = (v_perturbed[2::3] + v_perturbed[2::3][np.newaxis, :])**2
    overlap_constraints_perturbed = dist_sq - min_dist_sq
    overlap_constraints_perturbed = overlap_constraints_perturbed[i_indices, j_indices]
    cons_overlap_perturbed = [{"type": "ineq", "fun": lambda v, idx=idx: overlap_constraints_perturbed[idx]} for idx in range(len(overlap_constraints_perturbed))]

    cons_perturbed += cons_overlap_perturbed

    res_perturbed = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds_perturbed,
                             constraints=cons_perturbed, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_perturbed.x if res_perturbed.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())