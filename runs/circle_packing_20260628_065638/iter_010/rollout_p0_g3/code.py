import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering initialization
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    r0 = 0.35 / cols - 1e-3
    
    # Convert to bounded positions
    xs = (xs * cols - 0.5) / (cols - 1)
    ys = (ys * rows - 0.5) / (rows - 1)
    
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
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
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Radical reconfiguration: random geometric clustering
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        
        # Random geometric clustering
        new_xs = np.random.rand(n)
        new_ys = np.random.rand(n)
        new_xs = (new_xs * cols - 0.5) / (cols - 1)
        new_ys = (new_ys * rows - 0.5) / (rows - 1)
        
        perturbed_v = v.copy()
        perturbed_v[0::3] = new_xs
        perturbed_v[1::3] = new_ys
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion: expand the most tightly packed cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        
        # Calculate pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[0][i] - centers[0][j]
                dy = centers[1][i] - centers[1][j]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
                dists[j, i] = dists[i, j]
        
        # Identify the most tightly packed cluster
        min_dist = np.min(dists, axis=1)
        most_tightly_packed_idx = np.argmin(min_dist)
        
        # Expand its radius slightly and adjust its position to maintain feasibility
        v[3*most_tightly_packed_idx + 2] += 0.003
        v[3*most_tightly_packed_idx] += 0.005
        v[3*most_tightly_packed_idx+1] += 0.005
        
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())