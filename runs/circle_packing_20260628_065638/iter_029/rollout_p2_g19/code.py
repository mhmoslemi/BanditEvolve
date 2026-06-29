import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Base grid parameters and improved perturbation for dynamic spatial distribution
    grid_x = (np.arange(cols) + 0.5) / cols
    grid_y = (np.arange(rows) + 0.5) / rows
    
    # Generate diverse initial positions via randomized geometric tiling and staggered grid
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        
        # Base x/y coordinates from tiling
        x = grid_x[col]
        y = grid_y[row]
        
        # Add controlled perturbation based on row spacing to enable reconfiguration
        if row % 2 == 1:
            x += 0.45 / cols
        else:
            x += np.random.uniform(-0.16, 0.16) * (1.0 / cols)
        
        y += np.random.uniform(-0.2, 0.2) * (1.0 / rows)
        
        # Apply bounded clipping
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        y = np.clip(y, 1e-6, 1.0 - 1e-6)
        
        xs.append(x)
        ys.append(y)
    
    # Dynamic radius scaling improved with adaptive spatial distribution awareness
    base_radius = 0.34 / cols + 0.002 - 1e-3  # Tuned for better expansion
    r0 = base_radius * np.ones(n)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # Ensure bounds are exactly 3n entries with tighter radius lower bound
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-4, 0.495))  # Slightly tighter to prevent edge issues
    
    # Objective: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraints with tight tolerances and vectorization
    cons = []
    
    # Vectorized boundary constraints with explicit lambda and captured i
    for i in range(n):
        # Left constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized circle-circle overlap constraints with optimized formulation
    for i in range(n):
        for j in range(i + 1, n):
            # Use precomputed squared distance to avoid sqrt and enable vectorization
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                    - (v[3*i+2] + v[3*j+2])**2
                )
            })
    
    # Initial optimization with strict tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-10, "eps": 1e-10})
    
    # If not successful, apply asymmetric spatial perturbation for reconfiguration
    if not res.success:
        print("Initial optimization failed, initiating asymmetric spatial reconfiguration...")
        # Generate spatial hash with row/column-aware amplification
        spatial_hashes = np.random.rand(n, 2) * 0.05
        perturbed_v = v0.copy()
        for i in range(n):
            row = i // cols
            col = i % cols
            # Apply larger perturbation for low-density regions
            scale = np.sqrt((col + 1) / cols) * np.sqrt((row + 1) / rows) * 0.5
            dx_perturb = spatial_hashes[i, 0] * (r0[i] / np.mean(r0) * 1.2) * scale
            dy_perturb = spatial_hashes[i, 1] * (r0[i] / np.mean(r0) * 1.2) * scale
            perturbed_v[3*i] += dx_perturb
            perturbed_v[3*i+1] += dy_perturb
        
        # Re-run optimization on perturbed vector
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 800, "ftol": 1e-11, "gtol": 1e-10, "eps": 1e-10})
    
    # Targeted reconfiguration: find and expand least constrained sphere
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix and min distances calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate growth factor with spatial constraint awareness
        current_total = np.sum(radii)
        expansion_target = current_total + 0.011  # Aggressive expansion due to reconfiguration
        expansion_rate = (expansion_target - current_total) / (n-1)
        
        # Create expansion vector with adaptive amplification
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_rate * 1.3  # Aggressive expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Adaptively scale expansion based on spatial density and proximity
                dist_from_least = min_dists[i]
                spatial_factor = 1.0 + 0.4 * (np.linalg.norm(centers[i]) / np.std(centers))
                expansion_i = expansion_rate * (1.0 + 0.13 * np.random.rand()) * spatial_factor
                new_radii[i] += expansion_i
        
        # Apply expansion with rigorous validation
        iterations = 0
        max_iterations = 4
        while iterations < max_iterations:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion by 6% per failure
                new_radii = radii + (new_radii - radii) * 0.94
                iterations += 1
        
        # Update with expanded radii and refine
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-10, "eps": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())