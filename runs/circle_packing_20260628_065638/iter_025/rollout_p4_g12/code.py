import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    # Initialize with a staggered, randomized grid to promote even distribution
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        # Create base grid offset and stagger rows
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / cols
        # Randomized offset with increased variance for better exploration
        x = x_center + np.random.uniform(-0.12, 0.12)
        y = y_center + np.random.uniform(-0.12, 0.12)
        # Stagger even rows to prevent column alignment
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3
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
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Apply geometric hashing reconfiguration to induce non-local rearrangement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric hashing matrix with enhanced randomness and spatial hashing
        spatial_hash = np.random.rand(n, 2)
        perturbed_v = v.copy()
        for i in range(n):
            # Apply larger but directed perturbation for non-local reshuffling
            perturb_amount = np.random.uniform(0.015, 0.035)
            perturbed_v[3*i] += np.random.uniform(-perturb_amount, perturb_amount) * spatial_hash[i, 0]
            perturbed_v[3*i+1] += np.random.uniform(-perturb_amount, perturb_amount) * spatial_hash[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    # Apply targeted radius expansion to the circle with the smallest non-zero radius
    # while enforcing strict reordering of adjacency relationships
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        # Choose the circle with the largest minimum distance for expansion
        least_constrained_idx = np.argmax(min_dists)
        
        # Expand its radius while maintaining total sum constraints
        total_sum = np.sum(radii)
        # Ensure a small but non-zero expansion
        expansion_amount = 0.008
        target_total_sum = total_sum + expansion_amount
        expansion_per_c = (target_total_sum - total_sum) / (n - 1)
        
        # Apply expansion with controlled perturbation and adjacency reconfiguration
        # Ensure the most constrained circle is not expanded to maintain stability
        new_radii = radii.copy()
        # Apply larger expansion to the least constrained circle
        expansion = expansion_per_c * 1.35
        new_radii[least_constrained_idx] += expansion
        
        # Distribute expansion to the rest while applying small perturbations
        for i in range(n):
            if i != least_constrained_idx:
                # Apply a slight randomized expansion to others
                new_radii[i] += expansion_per_c * (1.0 + np.random.uniform(-0.2, 0.1))
        
        # Enforce adjacency constraint reordering via distance-based perturbation
        # Create a new distance matrix after expansion
        new_centers = centers.copy()
        new_centers[3*least_constrained_idx] += np.random.uniform(-0.015, 0.015)
        new_centers[3*least_constrained_idx+1] += np.random.uniform(-0.015, 0.015)
        # Rebuild distance matrix for validation
        dx = new_centers[:, np.newaxis, 0] - new_centers[np.newaxis, :, 0]
        dy = new_centers[:, np.newaxis, 1] - new_centers[np.newaxis, :, 1]
        new_dists = np.sqrt(dx**2 + dy**2)
        
        # Validate the new configuration to ensure no overlaps
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                if new_dists[i, j] < new_radii[i] + new_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Update decision vector with expanded radii
            v_new = v.copy()
            v_new[2::3] = new_radii
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
        else:
            # If invalid, return to previous state
            pass
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())