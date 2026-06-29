import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Initial placement using a dynamic grid with asymmetric spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Dynamic grid spacing to create asymmetric layout
        col_gap = 0.4 / cols
        row_gap = 0.4 / rows
        x_center = (col + 0.5) * col_gap + np.random.uniform(-0.04, 0.04)
        y_center = (row + 0.5) * row_gap + np.random.uniform(-0.04, 0.04)
        # Alternate row shifts to avoid symmetry
        if row % 2 == 1:
            x_center += 0.2 * col_gap
        xs.append(x_center)
        ys.append(y_center)
    
    # Introduce minimal initial radii with a dynamic starting value
    r0 = 0.35 / rows - 1e-4
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.45)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11})

    # Targeted reconfiguration with non-local geometric hashing and adjacency constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Create asymmetric hashing for spatial reconfiguration
        # Use adaptive hashing based on spatial density and radii
        spatial_hash = np.random.rand(n, 2) * 0.05 + (radii / np.mean(radii)) * 0.002
        adjacency_hash = np.random.rand(n, 2) * 0.04 + (radii / np.mean(radii)) * 0.001

        # Generate perturbed decision vector with spatial and adjacency-aware hashing
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
            # Apply directional adjacency expansion
            if i < n - 2:
                perturbed_v[3*i+2] += adjacency_hash[i, 0] * 0.004
                perturbed_v[3*i+1] += adjacency_hash[i, 1] * 0.002
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    # Targeted radius expansion on least constrained circle with directional bias
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate growth based on current total sum and potential for expansion
        # Target total expansion is 0.008 with gradient adjustment based on current configuration
        current_total = np.sum(radii)
        target_growth = 0.008
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))

        # Create expansion vector with targeted expansion on least constrained
        # Use directional hashing for enhanced perturbation
        directional_hash = np.random.rand(n, 2) * 0.04
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Apply directional expansion with spatial hashing and adjacency
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                expansion_i = expansion_factor * (1.0 + directional_hash[i, 0] * 0.5)
                if adj_weight < 0.1:
                    expansion_i *= 1.3  # Boost for nearby circles
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
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
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())