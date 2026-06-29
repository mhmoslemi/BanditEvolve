import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with grid-based distribution and small random perturbations
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols + np.random.uniform(-0.03, 0.03)
        y_center = (row + 0.5) / rows + np.random.uniform(-0.03, 0.03)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x_center += 0.5 / cols
        xs.append(x_center)
        ys.append(y_center)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using matrix operations
    v = v0.copy()  # Prepare vector for efficient distance calculation
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]
    
    # Precompute pairwise distances and distances squared for efficient constraint calculation
    dx = centers[:, 0, np.newaxis] - centers[:, np.newaxis, 0]
    dy = centers[:, 1, np.newaxis] - centers[:, np.newaxis, 1]
    dist_sq = dx**2 + dy**2
    sum_radii = radii[:, np.newaxis] + radii[np.newaxis, :]
    dist_sq_min = np.min(dist_sq, axis=0)
    constraint_idx = np.where(dist_sq < (sum_radii**2 - 1e-12))
    
    # Vectorized constraint function for all pair-wise constraints
    def constraint_func_overlap(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dx = centers[:, 0, np.newaxis] - centers[:, np.newaxis, 0]
        dy = centers[:, 1, np.newaxis] - centers[:, np.newaxis, 1]
        dist_sq = dx**2 + dy**2
        sum_radii = radii[:, np.newaxis] + radii[np.newaxis, :]
        return dist_sq - (sum_radii**2 - 1e-12)
    
    # Create constraints that enforce non-overlapping
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func_overlap(v)[i, j]})
    
    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-8})
    
    # Second optimization with perturbations
    if res.success:
        v = res.x
        perturbation = np.random.uniform(-0.02, 0.02, size=3 * n)
        perturbed_v = v + perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    
    # Final optimization
    if res.success:
        v = res.x
        perturbation = np.random.uniform(-0.01, 0.01, size=3 * n)
        perturbed_v = v + perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-8})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())