import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initial setup with spatial hashing to break symmetry
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add spatial hash for non-local reconfiguration
        x = x_center + np.random.uniform(-0.08, 0.08) + (np.random.rand() - 0.5) * 0.08
        y = y_center + np.random.uniform(-0.08, 0.08) + (np.random.rand() - 0.5) * 0.08
        if row % 2 == 1:
            x += 0.5 / cols
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

    # Vectorized overlap constraints using broadcasting
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization pass
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Apply geometric hashing for non-local reconfiguration and topological disruption
    if res.success:
        v = res.x
        perturb_scale = 0.1
        # Generate geometric hashes for non-local reconfiguration
        spatial_hash = np.random.rand(n, 2) * 2 * perturb_scale - perturb_scale
        perturbed_v = v.copy()
        
        # Apply perturbation
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Recompute constraints with new centers
        # Recompute all constraints with new perturbed centers
        def get_overlap_constraints(v):
            cons = []
            for i in range(n):
                # Boundary constraints
                cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
                cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
                cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
                cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            # Overlap constraints
            for i in range(n):
                for j in range(i + 1, n):
                    def constraint_func(v, i=i, j=j):
                        dx = v[3*i] - v[3*j]
                        dy = v[3*i+1] - v[3*j+1]
                        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                    cons.append({"type": "ineq", "fun": constraint_func})
            return cons
        
        # Re-optimizing with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=get_overlap_constraints(perturbed_v), options={"maxiter": 500, "ftol": 1e-11})

    # Targeted expansion on least constrained circle with adjacency-aware constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Apply controlled expansion to least constrained circle
        target_total_sum = np.sum(radii) + 0.006
        expansion_factor = (target_total_sum - np.sum(radii)) / n
        
        # Create new radii with expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        
        # Apply expansion to other circles with minor stochastic variations
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())  # slight variation
                new_radii[i] += expansion_i
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=get_overlap_constraints(v_new), options={"maxiter": 500, "ftol": 1e-11})

    # Final validation and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # clip to avoid numerical instability
    return centers, radii, float(radii.sum())