import numpy as np

def run_packing():
    n = 26
    
    # Initialize positions using a hybrid geometric hashing + staggered grid approach
    cols = 5
    rows = (n + cols - 1) // cols
    xs = []
    ys = []
    
    # Step 1: Base grid initialization with geometric hashing applied to random offsets
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Apply multi-level geometric hashing for more complex spatial patterns
        hash_seed = np.random.RandomState(i * 17).randn()
        x_offset = (base_x * (hash_seed % 0.1)) + (base_x * (hash_seed % 0.01))
        y_offset = (base_y * (hash_seed % 0.1)) + (base_y * (hash_seed % 0.01))
        
        # Add staggered row offset
        if row % 2 == 1:
            x_offset += 0.5 / cols
        
        x = base_x + x_offset
        y = base_y + y_offset
        xs.append(x)
        ys.append(y)
    
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

    # Vectorized overlap constraints using broadcasted operations for better performance
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    initial_res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                          constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply radical geometric hashing reconfiguration
    if initial_res.success:
        v = initial_res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Create geometric hash with multiple levels of randomization
        hash_seeds = np.random.RandomState(np.random.randint(0, 10**8)).randn(n, 5)
        hash_offsets = np.zeros((n, 2))
        
        for i in range(n):
            # First level: small base jitter
            hash_offsets[i, 0] = hash_seeds[i, 0] * 0.01
            hash_offsets[i, 1] = hash_seeds[i, 1] * 0.01
            
            # Second level: larger spatial offset
            hash_offsets[i, 0] += hash_seeds[i, 2] * 0.02
            hash_offsets[i, 1] += hash_seeds[i, 3] * 0.02
            
            # Third level: spatial distortion
            hash_offsets[i, 0] += hash_seeds[i, 4] * 0.01 * np.sin(2 * np.pi * (i * 0.1))
            hash_offsets[i, 1] += hash_seeds[i, 4] * 0.01 * np.cos(2 * np.pi * (i * 0.1))
        
        # Apply jitter to centers
        new_centers = centers + hash_offsets
        new_centers = np.clip(new_centers, 0.0, 1.0)
        
        # Create perturbed decision vector
        perturbed_v = v.copy()
        perturbed_v[0::3] = new_centers[:, 0]
        perturbed_v[1::3] = new_centers[:, 1]
        
        # Re-evaluate with new spatial configuration
        reconfigured_res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                                   constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
        
        if reconfigured_res.success:
            v = reconfigured_res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
        
        # Step 2: Identify the circle with the smallest non-zero radius
        sorted_indices = np.argsort(radii)
        smallest_radius_idx = sorted_indices[0]
        smallest_radius = radii[smallest_radius_idx]
        
        # Step 3: Apply controlled radius expansion to smallest radius circle while maintaining non-overlap
        if smallest_radius > 1e-6:
            # Calculate distance to all other centers for the smallest radius circle
            dx_min = centers[smallest_radius_idx, 0] - centers[:, 0]
            dy_min = centers[smallest_radius_idx, 1] - centers[:, 1]
            dist_min = np.sqrt(dx_min**2 + dy_min**2)
            
            # Find minimum distance to other circles for the smallest radius circle
            min_dist = np.min(dist_min[dist_min > 1e-12])
            
            # Calculate expansion factor based on current radius and available distance space
            expansion_factor = (min_dist - 2 * smallest_radius) / 2.0
            
            if expansion_factor > 0:
                # Create new radii with expansion to the smallest radius circle
                new_radii = radii.copy()
                new_radii[smallest_radius_idx] += expansion_factor
                new_radii = np.clip(new_radii, 1e-4, 0.45)
            
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
                v = v_new

    # Step 4: Final optimization with topological reordering of adjacency relationships
    final_res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                         constraints=cons, options={"maxiter": 600, "ftol": 1e-12})
    
    v = final_res.x if final_res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())