import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Create a grid layout with spatial distribution and random perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add controlled randomized jitter for diversification
        x = x_center + np.random.uniform(-0.05, 0.05) * (1.0 / rows)
        y = y_center + np.random.uniform(-0.05, 0.05) * (1.0 / cols)
        # Alternate rows for staggered pattern
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 / rows)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds list with consistent length and tight constraints
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Define objective to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Generate constraints with proper closures and vectorization
    cons = []

    # Boundary constraints for centers
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Overlap constraints with enhanced numerical stability
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorized distance squared constraint for efficient computation
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2})

    # First optimization pass with high accuracy and moderate iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-8})

    # Spatial reconfiguration with adaptive hashing for better exploration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash with adaptive scale for enhanced diversity
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        scaled_radii = radii / np.mean(radii) * 0.8  # Scaled for controlled perturbation
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * scaled_radii[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scaled_radii[i]
        
        # Re-optimize with new spatial layout
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})

    # Strategic reordering based on geometric constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify circles with smallest non-zero radius for expansion
        smallest_radius_idx = np.argmin(radii[radii > 1e-6])
        smallest_radius = radii[smallest_radius_idx]
        mean_radius = np.mean(radii)
        
        # Heuristic expansion with spatial awareness
        expansion_factor = (0.0075 / (n - 1)) * (mean_radius / np.sum(radii))
        
        # Create new radii with controlled expansion
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] = max(radii[smallest_radius_idx] + expansion_factor * 1.2, mean_radius * 0.95)
        for i in range(n):
            if i != smallest_radius_idx:
                new_radii[i] += expansion_factor * np.random.uniform(0.8, 1.2)

        # Refinement process with validation
        iterations = 0
        while iterations < 3:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Check for overlaps with tolerance
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
                # Scale down expansion if overlaps
                new_radii = radii + (new_radii - radii) * 0.95
                iterations += 1

        # Apply refined radii and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())