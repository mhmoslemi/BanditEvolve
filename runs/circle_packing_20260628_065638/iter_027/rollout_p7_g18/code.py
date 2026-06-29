import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with improved spatial diversity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Use more refined geometric grid with adaptive spacing
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add small, random offsets for spatial dispersion
        x_offset = np.random.uniform(-0.08, 0.08)
        y_offset = np.random.uniform(-0.08, 0.08)
        # Create staggered grid
        if row % 2 == 1:
            x_center += 0.25 / cols
        x = x_center + x_offset
        y = y_center + y_offset
        xs.append(x)
        ys.append(y)
    
    # Initial radius with improved starting point
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Boundary constraints with closure-based lambda handling
    cons = []
    for i in range(n):
        # Left bound: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right bound: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom bound: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top bound: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        
    # Overlap constraints with advanced vectorization
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda captures with fixed i and j to ensure correct indexing
            cons.append({
                "type": "ineq", 
                "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                      - (v[3*i+2] + v[3*j+2])**2
            })

    # First-stage optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply 'shake' heuristic if optimization succeeds
    if res.success:
        # Compute current radii and centers
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial shaking based on circle sizes
        shake_map = np.random.randn(n, 2) * (np.sqrt(radii) / np.sum(radii))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += shake_map[i, 0] * radii[i]
            perturbed_v[3*i+1] += shake_map[i, 1] * radii[i]
        
        # Re-evaluate perturbed parameters with enhanced tolerance
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    # Targeted expansion of smallest-radius circles with intelligent perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circles with minimal distance to others - potential for expansion
        min_dist = np.min(dists, axis=1)
        expansion_candidates = np.argsort(min_dist)  # Index of circles to expand
        
        # Initialize expansion vector
        new_radii = radii.copy()
        total_curr = np.sum(radii)
        target_growth = 0.008  # Aim for ~0.08% total increase
        growth_per_circle = (target_growth) / (n - 1)
        
        # Implement controlled expansion
        for idx in expansion_candidates:
            # Determine expansion amount based on spatial influence
            expansion_amount = growth_per_circle * (1.0 + 0.05 * np.random.rand())
            # Check if expansion will result in new overlaps
            new_radii[idx] += expansion_amount
            # Check if expansion is feasible
            while True:
                expanded_centers = np.column_stack([v[0::3], v[1::3]])
                expanded_radii = new_radii.copy()
                
                # Validate new configuration
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                        dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                        if dist_exp < expanded_radii[i] + expanded_radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    break
                else:
                    # If invalid, reduce expansion
                    new_radii[idx] -= 0.1 * expansion_amount

        # Update the decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization stage with enhanced constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Final refinement and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())