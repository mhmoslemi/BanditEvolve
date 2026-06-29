import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initial randomized geometric hashing with spatial perturbation
    xs = []
    ys = []
    rand_shift = np.random.rand(n, 2) * 0.04
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + rand_shift[i, 0] - 0.02
        y = y_center + rand_shift[i, 1] - 0.02
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
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

    # Vectorized constraints for boundaries with tighter tolerance
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing
    # We'll use vectorized pairwise distance calculation
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    # Topological disruption: random geometric hashing for reconfiguration
    if res.success:
        v = res.x
        # Introduce asymmetric spatial perturbation with directionality
        rand_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += rand_hash[i, 0] * (0.5 if i % 2 == 0 else -1.0)
            perturbed_v[3*i+1] += rand_hash[i, 1] * (0.5 if i % 2 == 0 else -1.0)
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    # Targeted expansion with adjacency-aware and minimal distance-based selection
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dists = np.zeros((n, n))
        for i in range(n):
            dx = centers[i, 0] - centers
            dy = centers[i, 1] - centers
            dists[i] = np.sqrt(dx*dx + dy*dy)
        
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        smallest_radius_idx = np.argmin(radii)
        total_sum = np.sum(radii)
        
        # Calculate expansion factor and target new configuration
        expansion_factor = 0.01 / (n - 1)  # Controlled expansion
        new_radii = radii.copy()
        # Expand least constrained and smallest circle more
        new_radii[least_constrained_idx] += expansion_factor * 1.5
        new_radii[smallest_radius_idx] += expansion_factor * 1.5
        # Expand all other circles
        for i in range(n):
            if i != least_constrained_idx and i != smallest_radius_idx:
                new_radii[i] += expansion_factor
        
        # Apply adjacency constraint to enforce new layout
        # We'll create a new vector with adjusted radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Final cleanup
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())