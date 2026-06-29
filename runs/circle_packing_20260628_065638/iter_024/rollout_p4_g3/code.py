import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    # Initialize with randomized clustered tiling for better convergence
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Introduce spatial clustering and randomness to avoid symmetry
        # Add Gaussian-like perturbation to create varied spatial distribution
        x = base_x + np.random.normal(0, 0.05)
        y = base_y + np.random.normal(0, 0.05)
        # Create staggered grid for better packing efficiency
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimate using area-based packing and geometric constraints
    base_radius = 0.36 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, base_radius)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Must match 3*n length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: lambda _: 1.0 - v[3*i] - v[3*i+2])()})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: lambda _: v[3*i] - v[3*i+2])()})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: lambda _: 1.0 - v[3*i+1] - v[3*i+2])()})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: lambda _: v[3*i+1] - v[3*i+2])()})
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization with high precision and extended iteration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-9})

    # Trigger constrained topological shift for enhanced packing
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Create geometric hash for spatial reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.05
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += spatial_hash[i, 0]
            new_v[3*i+1] += spatial_hash[i, 1]
        
        # Reoptimize with new spatial configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-10, "gtol": 1e-9})

    # Targeted radius expansion based on spatial constraints
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.zeros((n, n))
        
        # Efficient vectorized distance calculation
        for i in range(n):
            dx = centers[:, 0] - centers[i, 0]
            dy = centers[:, 1] - centers[i, 1]
            dists[i, :] = np.sqrt(dx**2 + dy**2)
        
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argsort(min_dists)[-1]  # Least constrained circle
        
        # Calculate expansion factor with total sum constraint
        target_total_sum = np.sum(radii) + 0.008  # Controlled expansion
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Adjust radii while preserving spatial relationships
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * 0.9
        
        # Reevaluate with new radii and spatial configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-10})

    # Final clean-up and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())