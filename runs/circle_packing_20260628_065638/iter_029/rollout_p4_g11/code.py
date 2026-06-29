import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Enhanced initialization: geometric lattice with stochastic perturbation and spatial diversity
    xs = []
    ys = []
    for i in range(n):
        # Column and row indexing
        col = i % cols
        row = i // cols
        # Base positions with adaptive scaling (larger grid has smaller spacing)
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Add spatial randomization based on grid spacing: more variance for denser grids
        x_rand_factor = np.clip(0.1 / cols, 0.01, 0.08)
        y_rand_factor = np.clip(0.1 / rows, 0.01, 0.08)
        # Use normal distribution for smoother perturbation
        x_pert = np.random.normal(0, x_rand_factor)
        y_pert = np.random.normal(0, y_rand_factor)
        # Alternating rows to simulate staggered grid
        if row % 2 == 1:
            x_pert += np.random.uniform(-0.005 * cols, 0.005 * cols)  # Row-based lateral shift
        x = base_x + x_pert
        y = base_y + y_pert
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation optimized for spatial distribution
    # Radius is proportional to square root of spacing, but with more variance for denser grids
    base_spacing = min(1.0 / cols, 1.0 / rows) * 0.9
    r0 = 0.7 * base_spacing / cols - 1e-3  # Slightly tighter than previous heuristic
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Bounds configuration: ensure consistent length for optimization vector
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.45)]  # tighter min radius, lower max radius

    # Objective function optimized for scalability and stability
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Negative of sum to optimize with minimization

    # Optimized constraint formulation: avoid lambda capture issues
    cons = []

    # Boundary constraints reformulated using broadcasting
    # Each circle has 4 constraints: x >= r, x + r <= 1, y >= r, y + r <= 1
    for i in range(n):
        # x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints: use vectorized computation with explicit i,j indexing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                # Vectorized distance squared minus sum of radii squared
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with advanced solver parameters
    # Use L-BFGS-B for better performance in high-dimensional spaces
    res = minimize(neg_sum_radii, v0, method="L-BFGS-B", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-9,
                                             "disp": False, "maxcor": 100, "early_stopping": True})

    # Post-optimization spatial reconfiguration for edge cases
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Find most constrained circle using spatial analysis
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.hypot(dx, dy)
                dists[j, i] = dists[i, j]
        
        # Find the circle with the smallest minimum distance to other circles
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Most confined
        
        # Create a dedicated growth area for most constrained circle
        max_growth_ratio = 0.02  # Allow for 2% growth of the most constrained circle
        base_growth = max_growth_ratio * radii[least_constrained_idx] * 0.8

        # Apply spatial hashing to generate new perturbations
        # Use relative position scaling based on grid
        spatial_hash = np.random.rand(n, 2) * 0.12  # Larger hash space to avoid collapse
        for i in range(n):
            # Perturb only the most constrained circle with targeted expansion
            if i != least_constrained_idx:
                v[3*i] += spatial_hash[i, 0] * (radii[i]/np.mean(radii)) 
                v[3*i+1] += spatial_hash[i, 1] * (radii[i]/np.mean(radii)) 
            else:
                v[3*i] += spatial_hash[i, 0] * (radii[i]/np.mean(radii)) * 0.5
                v[3*i+1] += spatial_hash[i, 1] * (radii[i]/np.mean(radii)) * 0.5
                # Apply targeted expansion to most constrained circle
                v[3*i+2] += base_growth  # Growth based on proportionality

        # Re-evaluate with new configuration in a lower iteration step to avoid overshoot
        res = minimize(neg_sum_radii, v, method="L-BFGS-B", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-9})

    # Final spatial validation with adaptive expansion based on geometric hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Perform a secondary refinement using geometric hashing for spatial diversity
        # Generate a set of candidate positions for each circle using a geometric hashing grid
        hash_grid = np.random.rand(n, 2) * 0.08  # Smaller hash space to avoid collapse
        for i in range(n):
            # Apply a small displacement for diversity
            v[3*i] += hash_grid[i, 0] * (radii[i]/np.mean(radii)) * 0.5
            v[3*i+1] += hash_grid[i, 1] * (radii[i]/np.mean(radii)) * 0.5
        
        # Re-evaluate with new hashing configuration
        res = minimize(neg_sum_radii, v, method="L-BFGS-B", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-9})

    # Final cleanup and validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.45)  # Clamp min and max radius
    return centers, radii, float(radii.sum())