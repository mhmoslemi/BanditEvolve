import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initial positions with randomized staggered grid and spatial hashing
    
    xs = []
    ys = []
    
    # Create a base grid with staggered rows
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add spatial hashing to break symmetry
        spatial_hash = np.random.rand(2) * 0.1 - 0.05
        x = x_center + spatial_hash[0]
        y = y_center + spatial_hash[1]
        
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        
        xs.append(x)
        ys.append(y)
    
    # Calculate baseline radii based on grid spacing
    base_radius = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, base_radius)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length matches 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Generate constraints with vectorized function binding
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with pre-compiled function binding
    for i in range(n):
        for j in range(i+1, n):
            # Precompute function binding to avoid lambda closure issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tight tolerances and early reconfiguration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-10})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply randomized geometric hashing for spatial reconfiguration
        # Create a dynamic hash grid to perturb configurations
        hash_grid = np.random.rand(n, 2) * 0.1 - 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_grid[i, 0]
            perturbed_v[3*i+1] += hash_grid[i, 1]
        
        # Re-evaluate with new perturbed centers
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Identify least constrained circle with vectorized distance calculation
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Apply directional expansion with spatial hashing and adjacency reordering
        expansion_max = 0.012  # Higher than predecessor
        expansion_factor = 0.95 * (expansion_max / (n - 1))  # Slightly reduced
        
        # Create directional expansion vector with spatial hashing influence
        hash_direction = np.random.rand(n, 2) * 0.2 - 0.1
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Expand based on directional hash for adjacency reordering
                new_radii[i] += expansion_factor * (1.0 + hash_direction[i, 0] * 0.4)
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = np.clip(new_radii, 1e-6, 0.5)
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate configuration with full overlap check
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion if invalid
                new_radii = radii + (new_radii - radii) * 0.98
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization on the expanded configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())