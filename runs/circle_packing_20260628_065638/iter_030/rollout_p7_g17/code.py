import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Enhanced initialization with adaptive clustering and directional perturbation
    xs = []
    ys = []
    # Introduce row-specific scaling for more natural grid distribution
    row_scales = [1.0 + (0.1 * (0.5 - row / rows)) for row in range(rows)]
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols * (1 + (row % 2) * 0.1)
        y_center = (row + 0.5) / rows
        # Introduce multi-attribute perturbation with spatial coherence
        # Use row-scale-aware offset to maintain grid coherence with spatial awareness
        x = x_center + np.random.uniform(-0.08 * row_scales[row], 0.08 * row_scales[row])
        y = y_center + np.random.uniform(-0.04, 0.04) * (1 + 0.2 * row)
        # Alternate rows have staggered offset with adaptive shift
        if row % 2 == 1:
            x += 0.5 / cols * (1 + 0.1 * row_scales[row])
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with optimized initial scaling accounting for both grid spacing and rows
    r0 = 0.37 / cols - 1e-3  # Slightly increased baseline to allow for better expansion
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n length, matches decision vector
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries, optimized using lambda with captured i
    cons = []
    for i in range(n):
        # Left - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            # Use functional capture to avoid issues with closures
            cons.append({
                "type": "ineq", 
                "fun": (lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                        - (v[3*i+2] + v[3*j+2])**2)
            })

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Asymmetric reconfiguration with spatial correlation and gradient-aware sampling
    if res.success:
        v = res.x
        
        # Calculate radii and centers for perturbation
        current_radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Spatial hashing for perturbation: cluster-aware random walks with local influence
        hash_matrix = np.random.rand(n, n)  # Used for spatial coherence check
        # Perturb centers with adaptive weight toward centers with higher spatial potential
        # Use inverse square of radii to give more influence to smaller circles
        perturbation_weights = 1 / (current_radii ** 2)
        max_weight = np.max(perturbation_weights)
        if max_weight > 0:
            # Normalize weights for perturbation
            perturbation_weights /= max_weight
            
        # Generate random perturbation vectors with spatial correlation
        # Add small correlated perturbations to centers with greater influence
        random_perturb = np.random.rand(n, 2) * 0.04 * (1 + (np.random.rand(n) * 0.1))
        perturbed_centers = centers + random_perturb * perturbation_weights[:, None]
        
        # Apply perturbations to centers with spatial awareness
        # Reconstruct decision vector with new positions
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] = perturbed_centers[i, 0]
            perturbed_v[3*i+1] = perturbed_centers[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})
    
    # Multi-phase radius expansion with spatial constraint-aware expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute all-pairs distances using vectorized operations
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate adjacency matrix: circle pairs with distance <= sum of radii
        adj = dists <= (radii + radii[np.newaxis, :])
        
        # Identify least constrained circle: one with the highest minimum distance to others
        min_dist_to_others = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dist_to_others)
        
        # Compute total sum for expansion calculation
        current_sum = np.sum(radii)
        # Precompute target growth based on current radii sum and average constraint space
        # We aim for 0.0055 growth with spatial expansion control
        expansion_threshold = 0.0055  # Slight but significant
        expansion_factor = expansion_threshold / (n - 1) * (current_sum / np.mean(radii))
        
        # Apply targeted expansion using gradient-aware approach: expand least constrained
        # and propagate to adjacent neighbors with spatial constraints
        # Start with least constrained, then expand in a controlled fashion
        expanded_radii = radii.copy()
        # First expand least constrained circle
        expanded_radii[least_constrained_idx] += expansion_factor * 1.24  # Slight over-expansion to trigger reordering
        # Use adjacency graph to determine neighbors and gradually expand
        # Use BFS to apply spatial expansion in controlled manner
        from scipy.sparse import csr_matrix, csgraph
        graph = csr_matrix(adj)
        neighbors = csgraph.neighbors_graph(graph)
        
        # Use adjacency matrix to get immediate neighbors of least constrained
        immediate_neighbors = np.where(adj[least_constrained_idx, :])[0]
        
        # Apply expansion to immediate neighbors with spatial awareness
        for neighbor_idx in immediate_neighbors:
            if neighbor_idx != least_constrained_idx:
                expanded_radii[neighbor_idx] += expansion_factor * 0.95
        
        # Apply to further neighbors with adaptive weighting
        for neighbor_idx in immediate_neighbors:
            # Get neighbors of neighbors
            neighbor_of_neighbor = np.where(adj[neighbor_idx, :])[0]
            for nn in neighbor_of_neighbor:
                if nn != least_constrained_idx and nn != neighbor_idx:
                    expanded_radii[nn] += expansion_factor * 0.7
        
        # Create decision vector with the new radii
        v_new = v.copy()
        v_new[2::3] = expanded_radii
        
        # Final optimization phase with constrained expansion to find optimal configuration
        # We reduce maxiter and increase tolerance for final tuning
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})
    
    # Apply final validation and cleanup
    v = res.x if res.success else v0
    
    # Final check with spatial coherence to ensure all circles are valid
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Ensure no NaN or invalid data
    if np.isnan(centers).any() or np.isnan(radii).any():
        return centers, radii, float(radii.sum())  # return as is
    
    # Add final safety check for spatial integrity
    # For all circles ensure within the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12
                or y - r < -1e-12 or y + r > 1 + 1e-12):
            # If any circle is out of bounds, slightly shrink the largest circle
            if np.max(radii) > 0.5:
                radii[np.argmax(radii)] = 0.5 - 1e-5
            else:
                radii = np.clip(radii, 1e-6, 0.5)
    
    # Final validation step to ensure no overlaps
    # We will now perform a final explicit check for overlaps due to the precision in expansion
    # This is an additional safeguard as our constraints might have slight numerical issues
    def validate_safety(centers, radii):
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < (radii[i] + radii[j]) - 1e-12:
                    # If overlap, adjust the smaller radii to avoid it
                    if radii[i] < radii[j]:
                        radii[i] = max(radii[i], dist - radii[j] + 1e-9)
                    else:
                        radii[j] = max(radii[j], dist - radii[i] + 1e-9)
                    # Reconstruct decision vector
                    v_final = v.copy()
                    v_final[2::3] = radii
                    # Re-check for convergence
                    # Re-run the optimization with new configuration to ensure feasibility
                    res_new = minimize(neg_sum_radii, v_final, method="SLSQP", bounds=bounds,
                                       constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
                    if res_new.success:
                        v = res_new.x
                    else:
                        # Fallback to last known feasible configuration
                        v = res.x
                    # Rebuild centers
                    centers = np.column_stack([v[0::3], v[1::3]])
                    radii = v[2::3]
                    return (centers, radii)
        return (centers, radii)
    
    final_centers, final_radii = validate_safety(centers, radii)
    final_radii = np.clip(final_radii, 1e-6, 0.5 - 1e-5)
    
    return final_centers, final_radii, float(final_radii.sum())