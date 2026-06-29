import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a dual-layer stochastic grid with adaptive spatial perturbation
    # First layer: base staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.15, 0.15)  # Expanded range for diversity
        y = y_center + np.random.uniform(-0.15, 0.15)
        # Introduce subtle row-based asymmetry for more efficient packing
        if row % 3 == 1:
            x += np.random.uniform(-0.03, 0.03)
        xs.append(x)
        ys.append(y)
    
    r0 = (0.6 / cols) - 1e-3  # Higher initial radius for improved exploration
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n length, 26*3=78 entries

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective: maximize radius sum

    # Vectorized constraints: boundary and non-overlap
    cons = []
    for i in range(n):
        # Left boundary constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right boundary constraint: 1 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom boundary constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top boundary constraint: 1 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Overlap constraints using vectorized expressions
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i + 1] - v[3*j + 1])**2 
                         - (v[3*i + 2] + v[3*j + 2])**2})
    
    # Initial optimization: aggressive maxiter + tight ftol for convergence
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-8}
    )

    # First-phase reconfiguration: targeted local search with perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Adaptive spatial perturbation with radius-dependent scaling
        # Use a low-entropy spatial seed for deterministic reconfiguration
        np.random.seed(42)
        spatial_hash = np.random.rand(n, 2) * 5.0
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii) ** 0.8)
            perturbed_v[3*i + 1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii) ** 0.8)
        
        # Reconfigure using hybrid strategy: moderate maxiter with tighter tolerances
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11}
        )

    # Second-phase reconfiguration: adaptive expansion of "least constrained" circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial influence map with geometric distance-based weighting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2) + 1e-8
        
        # Influence map: inverting normalized min distances for "least constrained"
        min_dists = np.min(dists, axis=1)
        influence_map = 1.0 / (min_dists + 1e-8)
        influence_sum = np.sum(influence_map, axis=1)
        normalized_influence_map = influence_map / (influence_sum[:, np.newaxis] + 1e-8)
        least_constrained_idx = np.argmin(normalized_influence_map)  # Smallest influence = most room
        
        current_total = np.sum(radii)
        # Define a controlled aggressive expansion factor
        expansion_factor = 0.012 * (current_total / np.sum(radii)) ** 0.85
        
        # Apply targeted expansion with random perturbation to escape local optima
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Over-expansion to test edges
        for i in range(n):
            if i != least_constrained_idx:
                # Randomized expansion for diverse configurations
                noise = np.random.uniform(0.85, 1.05)  # 10% stochastic variation
                new_radii[i] += expansion_factor * noise
        
        # Validate expansion with vectorized collision detection
        def validate_expanded(new_radii):
            expanded_centers = np.column_stack([v[0::3], v[1::3]])
            expanded_radii = new_radii
            
            # Vectorized pairwise distance matrix (faster than nested loops)
            dx = expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0]
            dy = expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1]
            dist_matrix = np.sqrt(dx**2 + dy**2)
            
            # Check for overlaps using matrix-wise comparison
            for i in range(n):
                for j in range(i+1, n):
                    if dist_matrix[i, j] < expanded_radii[i] + expanded_radii[j] - 1e-12:
                        return False
            return True
        
        # Iteratively scale down expansion until valid configuration
        while True:
            # Create expanded vector
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            
            # Validate expanded configuration
            if validate_expanded(new_radii):
                break
            else:
                # Linearly interpolate towards original radii to escape invalid state
                new_radii = radii + (new_radii - radii) * 0.965
        
        # Final re-optimization using the refined configuration
        res = minimize(
            neg_sum_radii, 
            expanded_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11}
        )
        
        # Final validation check after optimization with tighter safety margin
        if not res.success:
            # Fallback to initial configuration when optimization fails
            res = minimize(
                neg_sum_radii, 
                v, 
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 200, "ftol": 1e-9}
            )
    
    # Final validation fallback: if all else fails, return best found configuration
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    # Clip radii to ensure positive and within bounds
    radii = np.clip(v[2::3], 1e-6, 0.5)
    return centers, radii, float(radii.sum())