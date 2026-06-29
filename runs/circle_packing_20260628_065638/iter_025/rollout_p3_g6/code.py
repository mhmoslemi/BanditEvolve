import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with adaptive staggered grid with optimized spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Use larger initial radii to allow for more room during reconfiguration
    r0 = 0.4 / cols - 1e-3
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})

    if res.success:
        v = res.x
        
        # Apply geometric hashing for topological reconfiguration
        # Use a more aggressive hash to create a new spatial configuration
        spatial_hash = np.random.rand(n, 2) * 0.15 - 0.075
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-construct constraints with new centers
        def get_overlap_constraints(v):
            cons = []
            for i in range(n):
                cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
                cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
                cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
                cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
            for i in range(n):
                for j in range(i + 1, n):
                    def constraint_func(v, i=i, j=j):
                        dx = v[3*i] - v[3*j]
                        dy = v[3*i+1] - v[3*j+1]
                        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                    cons.append({"type": "ineq", "fun": constraint_func})
            return cons
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=get_overlap_constraints(perturbed_v), options={"maxiter": 600, "ftol": 1e-12})
    
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute distances and identify the least constrained circle
        # Using vectorized broadcasting for efficiency
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify the circle with the smallest minimum distance to neighbors
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)
        
        # Compute expansion factor: prioritize the smallest circle
        expansion_factor = 0.006 / (n - 1)  # Controlled global expansion
        # Apply expansion to the least constrained circle with over-shooting
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * 0.9
        
        # Apply expansion and validate
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        # Create a temporary variable to store expanded centers
        expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
        # Validate expansion
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            v = expanded_v
        else:
            # Revert to last valid configuration if expansion causes overlap
            v = res.x
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())