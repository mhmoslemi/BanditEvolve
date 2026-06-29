import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add small randomized positional offset to break symmetry
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        # Staggered grid by offsetting alternate rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.36 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for the optimization
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Minimize negative of sum to maximize sum

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left boundary constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate the squared distance minus sum of radii squared as a constraint
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased iterations and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1700, "ftol": 1e-10, "gtol": 1e-9})

    # Shake the 5 smallest circles to escape shallow local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Identify the 5 smallest circles 
        smallest_indices = np.argsort(radii)[:5]

        # Compute their gradients based on surrounding circles
        grad_perturbation = np.zeros(3 * n)
        for idx in smallest_indices:
            # Add random spatial perturbations with gradient-aware randomness
            grad_perturbation[3*idx] += np.random.uniform(-0.04, 0.04)
            grad_perturbation[3*idx+1] += np.random.uniform(-0.04, 0.04)
            grad_perturbation[3*idx+2] += np.random.uniform(-0.003, 0.003)

        # Perturb the configuration and re-optimze
        perturbed_v = v + grad_perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-9})

    # Targeted reconfiguration by identifying least constrained circles - optimized method
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Vectorized pairwise distances
        dists = np.zeros((n, n))
        dx = centers[:, 0, np.newaxis] - centers[:, np.newaxis, 0]
        dy = centers[:, 1, np.newaxis] - centers[:, np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circles by looking at minimum distances
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argsort(min_dists)[-3:]  # Select top 3 least constrained
        
        # Expand radii of least constrained circles carefully
        # Calculate expansion factor
        total_sum = np.sum(radii)
        expansion_factor = 0.01 / (n - 1)
        
        # Adjust radii
        new_radii = radii.copy()
        for i in least_constrained_idx:
            new_radii[i] += expansion_factor * 1.3  # Over-expansion to trigger new configuration
            # Propagate expansion to neighbors to unlock new possibilities
            for j in np.arange(n):
                if j != i:
                    new_radii[j] += expansion_factor * 0.7
        
        # Update decision vector and re-optimze
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-9})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())