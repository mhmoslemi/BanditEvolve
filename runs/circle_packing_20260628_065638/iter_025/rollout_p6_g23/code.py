import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with more refined randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.12, 0.12)
        y = y_center + np.random.uniform(-0.12, 0.12)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        # Add slight perturbation to avoid initial symmetry
        x += np.random.uniform(-0.03, 0.03)
        y += np.random.uniform(-0.03, 0.03)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.32 / cols - 1e-3  # More aggressive initial radius to allow expansion
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2200, "ftol": 1e-12, "eps": 1e-10})

    # Apply stochastic shaking of small circles to escape local optima
    if res.success:
        v = res.x
        # Identify smallest circles for targeted shaking
        radii = v[2::3]
        small_mask = radii < np.mean(radii)  # Mark circles significantly smaller than average
        perturbation = 0.05  # Small perturbation strength
        # Apply perturbations to small circles only
        perturbed_v = v.copy()
        for i in range(n):
            if small_mask[i]:
                # Perturb position with small random shift
                perturbed_v[3*i] += np.random.uniform(-perturbation, perturbation)
                perturbed_v[3*i+1] += np.random.uniform(-perturbation, perturbation)
                # Clamp positions to square bounds
                perturbed_v[3*i] = max(0, min(1, perturbed_v[3*i]))
                perturbed_v[3*i+1] = max(0, min(1, perturbed_v[3*i+1]))
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-10})

    # Perform targeted expansion on circles with least spatial constraint
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate minimum distance for each circle
        min_dists = np.min(dists, axis=1)
        # Find the least constrained circle (maximum minimum distance)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate the base expansion factor based on current radii and spatial constraint
        base_radius = np.mean(radii)
        expansion_factor = (0.01) / (n - 1)  # Conservative base expansion target
        
        # Construct expansion vector with asymmetric increase
        new_radii = radii.copy()
        # Increase the least constrained circle more for spatial optimization
        new_radii[least_constrained_idx] += expansion_factor * 1.3
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Clamp radii to prevent overlap beyond constraint tolerance due to expansion
        new_radii = np.clip(new_radii, 1e-6, 0.5)

        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
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
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-10})

    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())