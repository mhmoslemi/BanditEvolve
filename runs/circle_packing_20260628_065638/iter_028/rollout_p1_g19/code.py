import numpy as np

def run_packing():
    n = 26
    # Geometric tiling with dynamic grid for better density
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Initialize positions using adaptive grid tiling and asymmetric stagger
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base tile grid center
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Asymmetric staggered offset for reduced symmetry (based on column parity)
        x_offset = np.random.uniform(-0.08, 0.08) + (col % 3) * (0.03 if col % 2 == 0 else -0.03)
        y_offset = np.random.uniform(-0.08, 0.08) + (row % 3) * (0.03 if row % 2 == 0 else -0.03)
        x = base_x + x_offset
        y = base_y + y_offset
        xs.append(x)
        ys.append(y)
    # Initialize radii with higher base and more adaptive distribution
    r0 = 0.38 / cols - 1e-3  # Slightly higher base radius
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Same length as 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint generation
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
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with aggressive settings and early warm-up
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "disp": False})

    # Radical non-local geometric reconfiguration using spatial hashing and cluster detection
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Generate dense spatial hashing with adaptive scaling factor based on cluster density
        spatial_hash = np.random.rand(n, 2) * (0.07 ** (1/(np.sqrt(np.sum(radii**2)))))  
        perturbed_v = v.copy()
        for i in range(n):
            # Spatial hashing with scaled perturbation
            perturbed_v[3*i] += spatial_hash[i, 0] * (1 + 0.05 * np.random.rand()) * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (1 + 0.05 * np.random.rand()) * (radii[i] / np.mean(radii))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "disp": False})

    # Targeted radius expansion on least constrained circle with soft constraints and dynamic constraint reordering
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
        current_total = np.sum(radii)
        target_growth = 0.0085  # Slightly increased from previous 0.006
        expansion_factor_base = target_growth / (n - 1)
        # Apply dynamic expansion based on spatial hashing and adjacency to least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor_base * 1.5  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Calculate directional influence based on spatial hashing and neighbor proximity
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                directional_coeff = np.random.rand() * 0.3  # Small random spatial influence
                # Use exponential falloff for adjacency distance
                dist_scaling = np.exp(-adj_weight / (np.mean(radii) * 2))
                expansion = expansion_factor_base * dist_scaling * (1 + directional_coeff)
                new_radii[i] += expansion
        
        # Apply expansion with constraint validation and soft annealing
        # Start with 50% of the new radii for stability
        annealing_factor = 0.5
        new_radii = radii + (new_radii - radii) * annealing_factor
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
                # If invalid, decrease expansion slightly but maintain a minimum threshold
                new_radii = radii + (new_radii - radii) * 0.95
                annealing_factor *= 0.98
        
        # Update decision vector with reconfigured positions and expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and reconfigured spacing
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "disp": False})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())