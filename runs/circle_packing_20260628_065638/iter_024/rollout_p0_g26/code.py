import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with base grid and randomized geometric hashing
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / rows
    xs += np.random.uniform(-0.04, 0.04, n)
    ys += np.random.uniform(-0.04, 0.04, n)
    # Stagger alternate rows
    for i in range(n):
        if (i // cols) % 2 == 1:
            xs[i] += 0.5 / cols
    r0 = 0.34 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # x, y, radius

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints using functools.partial
    from functools import partial

    cons = []
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": partial(lambda v, i: v[3*i] - v[3*i+2], i=i)})
        # Right boundary
        cons.append({"type": "ineq", "fun": partial(lambda v, i: 1.0 - v[3*i] - v[3*i+2], i=i)})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": partial(lambda v, i: v[3*i+1] - v[3*i+2], i=i)})
        # Top boundary
        cons.append({"type": "ineq", "fun": partial(lambda v, i: 1.0 - v[3*i+1] - v[3*i+2], i=i)})

    # Vectorized overlap constraints using vector operations
    # Precompute pairwise distance squared and radius sum squared for vectorization
    def compute_overlap_constraints(v):
        centers = v.reshape((n, 3))[:, :2]
        radii = v.reshape((n, 3))[:, 2]
        dist_sq = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq[i, j] = dx * dx + dy * dy
        return dist_sq - (radii[:, np.newaxis] + radii[np.newaxis, :]) ** 2

    # Apply overlap constraints (inequality: dist^2 >= (r1 + r2)^2)
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: compute_overlap_constraints(v)[i, j]})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})

    # Apply randomized geometric hashing to escape local minima
    if res.success:
        v = res.x
        # Generate randomized geometric hashing vector
        hash_vec = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_vec[i, 0]
            perturbed_v[3*i+1] += hash_vec[i, 1]
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion using constraint tightness heuristic
    if res.success:
        v = res.x
        centers = v.reshape((n, 3))[:, :2]
        radii = v.reshape((n, 3))[:, 2]
        dist_sq = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq[i, j] = dx * dx + dy * dy
        
        # Calculate constraint tightness: distance to nearest neighbor
        min_dist = np.min(dist_sq, axis=1)
        min_radius = np.min(radii)

        # Find circle with largest minimum distance (least constrained)
        least_constrained_idx = np.argmax(min_dist)

        # Calculate expansion factor for controlled radius increase
        current_sum = np.sum(radii)
        expansion_factor = 0.01 / (n - 1)  # Safe base expansion factor

        # Apply expansion based on constraint margins
        for i in range(n):
            if i == least_constrained_idx:
                # Give extra boost to the least constrained circle
                v[3*i + 2] += expansion_factor * 1.3
            else:
                v[3*i + 2] += expansion_factor
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())