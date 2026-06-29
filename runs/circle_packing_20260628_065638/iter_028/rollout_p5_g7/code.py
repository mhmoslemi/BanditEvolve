import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Create initial positions with staggered grid and randomized offsets
    xs = np.zeros(n)
    ys = np.zeros(n)
    base_x_offsets = np.zeros(n)
    base_y_offsets = np.zeros(n)
    
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Introduce asymmetric randomized offsets
        base_x_offsets[i] = np.random.uniform(-0.1, 0.1)
        base_y_offsets[i] = np.random.uniform(-0.1, 0.1)
        
        xs[i] = base_x + base_x_offsets[i]
        ys[i] = base_y + base_y_offsets[i]
    
    # Add row-wise staggering
    for i in range(n):
        if (i // cols) % 2 == 1:
            xs[i] += 0.25 / cols
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure the bounds list has 3*n entries for the vector of length 3n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with better closures
    cons = []
    for i in range(n):
        # Left + radius <= 1
        def constraint_left(v, i=i):
            return 1.0 - v[3*i] - v[3*i+2]
        cons.append({"type": "ineq", "fun": constraint_left})
        # Right - radius >= 0
        def constraint_right(v, i=i):
            return v[3*i] - v[3*i+2]
        cons.append({"type": "ineq", "fun": constraint_right})
        # Bottom + radius <= 1
        def constraint_bottom(v, i=i):
            return 1.0 - v[3*i+1] - v[3*i+2]
        cons.append({"type": "ineq", "fun": constraint_bottom})
        # Top - radius >= 0
        def constraint_top(v, i=i):
            return v[3*i+1] - v[3*i+2]
        cons.append({"type": "ineq", "fun": constraint_top})
    
    # Vectorized overlap constraints with proper lambda capture
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_overlap(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_overlap})
    
    # First optimization with high iteration and tight tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-10})
    
    # Asymmetric spatial reconfiguration with adaptive stochastic scaling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Generate adaptive spatial perturbation with positional weight
        spatial_noise = np.random.rand(n, 2)
        spatial_perturbation = spatial_noise * 0.04 * (radii / np.mean(radii))
        
        # Apply adaptive spatial perturbation to the configuration
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_perturbation[i, 0]
            perturbed_v[3*i+1] += spatial_perturbation[i, 1]
        
        # Re-optimization with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute minimum distances for each circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion based on average distance and radii
        avg_dist = np.mean(min_dists)
        avg_radius = np.mean(radii)
        
        # Compute potential expansion and apply it with constraint validation
        expansion_factor = 0.4 * avg_dist / avg_radius
        max_allowed_expansion = 0.006
        
        # We'll do a controlled expansion with dynamic validation
        expansion = np.zeros(n)
        expansion[least_constrained_idx] = expansion_factor * 0.8  # slightly less to be safe
        expansion[expansion < 0] = 0.0
        
        for _ in range(3):
            # Tentative expansion
            new_radii = radii + expansion
            
            # Create perturbed configuration to test if expansion is viable
            perturb_scale = 0.05 * (new_radii / np.mean(new_radii))
            perturb = np.random.rand(n, 2) * perturb_scale
            
            perturbed_centers = centers + np.column_stack([perturb[:, 0], perturb[:, 1]])
            
            # Check for overlaps in new configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = perturbed_centers[i, 0] - perturbed_centers[j, 0]
                    dy = perturbed_centers[i, 1] - perturbed_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Accept expansion
                radii = new_radii
                break
            else:
                # Reduce expansion
                expansion = expansion * 0.8
        
        # Update our decision vector
        v_new = v.copy()
        v_new[2::3] = radii
        
        # Final optimization with adjusted radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())