import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with geometric tiling and dynamic spatial perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Introduce geometric tiling: scale by row to reduce cluster formation
        row_factor = 1.0 + 0.1 * np.random.rand()  # Introduce variation
        x_center *= row_factor
        x_center = np.clip(x_center, 0.05, 1 - 0.05)  # Ensure boundaries
        
        y_center *= 1.0 + 0.1 * np.random.rand()  # Introduce variation
        y_center = np.clip(y_center, 0.05, 1 - 0.05)
        
        # Apply non-uniform perturbation; more on the edges of the square
        perturbation_factor = np.random.rand() * 0.1
        x = x_center + np.random.uniform(-0.1 * perturbation_factor, 0.1 * perturbation_factor)
        y = y_center + np.random.uniform(-0.1 * perturbation_factor, 0.1 * perturbation_factor)
        
        # Shift alternate rows for spatial diversity
        if row % 2 == 1:
            x += 0.3 / cols * np.random.normal()
        
        xs.append(x)
        ys.append(y)
    
    # Set initial radii based on geometric spacing with tighter lower bound
    r0 = 0.4 * np.random.rand(n)  # Randomized base scaling
    r0[r0 < 1e-3] = 1e-3  # Ensure valid radii
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bound entries

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint setup with lambda closures
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    # Vectorized overlap constraint
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # First optimization run with increased iterations and improved tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "eps": 1e-12})

    # Radical geometric tiling-based reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply geometric tiling with adaptive scaling to spread out
        spatial_hash = (np.random.rand(n, 2) - 0.5) * 0.1 / (np.mean(radii) * 1.2)
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-12})

    # Targeted radius expansion on the smallest non-zero radius with dynamic target
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorize distances and find the least constrained circle
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Least constrained circle
        smallest_radius_idx = np.argmin(radii)
        
        # Use the smallest radius as the main expansion target
        target_idx = smallest_radius_idx
        
        # Calculate total sum and dynamic target expansion factor
        current_total = np.sum(radii)
        if current_total < 2.5:
            target_growth = 0.008  # Conservative expansion
        else:
            # Allow more aggressive growth for higher total sums
            target_growth = 0.012
        
        # Calculate expansion factor based on current radius and growth target
        expansion_factor = target_growth / (n - 1)
        
        # Create expansion vector with targeted expansion and stochastic spreading
        new_radii = radii.copy()
        new_radii[target_idx] += expansion_factor * 1.15  # Controlled over-expansion
        for i in range(n):
            if i != target_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())
        
        # Apply expansion with iterative constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
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
                # If invalid, reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final optimization run with even tighter tolerances
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())