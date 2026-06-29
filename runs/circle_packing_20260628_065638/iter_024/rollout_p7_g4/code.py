import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize positions with dynamic geometric hashing to break initial symmetry
    # Using a hybrid of grid and random perturbation with staggered offset
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Random offset for each circle to break uniformity
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Staggered grid for alternating rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)

    r0 = (0.5 / cols) * 0.9 - 1e-3  # Slightly smaller initial radius
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure bounds match 3n elements

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective: maximize sum of radii

    # Vectorized boundary constraint functions
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraint functions using geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-10})

    # First phase: local search perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        min_radius_idx = np.argmin(radii)  # Target the smallest radius for expansion

        # Apply small random perturbations to break local minima
        for _ in range(3):
            v = v.copy()
            v[3*min_radius_idx] += np.random.uniform(-0.02, 0.02)
            v[3*min_radius_idx+1] += np.random.uniform(-0.02, 0.02)
            v[3*min_radius_idx+2] += np.random.uniform(-0.002, 0.002)
            
            # Re-evaluate with adjusted parameters and higher tolerance
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons,
                           options={"maxiter": 200, "ftol": 1e-12, "eps": 1e-10})
            if res.success:
                v = res.x

    # Second phase: randomized geometric hashing to generate diverse configurations
    if res.success:
        v = res.x
        # Apply randomized geometric hashing to diversify positions
        random_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-9})

    # Third phase: target radius expansion on least constrained circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Compute pairwise distances for constraint assessment
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Identify circle with the largest minimum distance (least constrained)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate expansion factor to increase least constrained circle's radius
        total_sum = np.sum(radii)
        expansion_factor = 0.012 / (n - 1)  # Small controlled expansion

        # Adjust radii with adjacency-based expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.1  # Over-expansion to trigger reconfiguration
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor

        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii

        # Re-evaluate with new radii and configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-9})

    # Final check
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())