import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized staggered grid with refined spatial hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset with adaptive range based on proximity to edges
        x = x_center + np.random.uniform(-0.035 * (1 - np.abs(2 * col / cols - 1)), 
                                         0.035 * (1 - np.abs(2 * col / cols - 1)))
        y = y_center + np.random.uniform(-0.035 * (1 - np.abs(2 * row / rows - 1)), 
                                         0.035 * (1 - np.abs(2 * row / rows - 1)))
        # Alternate row shifting with adaptive offset
        if row % 2 == 1:
            x += 0.5 / cols * (1 - np.abs(2 * col / cols - 1))
        xs.append(x)
        ys.append(y)
    
    r0 = 0.33 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n matches v

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
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-11})
    
    # Disruptive geometric transformation with spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create adaptive geometric hash for reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            # Perturb based on proximity to edges and grid position
            edge_factor = 1.0 - np.max([v[3*i], 1.0 - v[3*i], v[3*i+1], 1.0 - v[3*i+1]])
            perturbed_v[3*i] += spatial_hash[i, 0] * (1 + 0.2 * edge_factor)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (1 + 0.2 * edge_factor)
        
        # Re-evaluate with new spatial config
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    # Targeted radius expansion with edge-aware constraint enforcement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Evaluate least constrained circle using edge-aware measure
        min_dists = np.min(dists, axis=1)
        weighted_min = min_dists * np.product([1.0 - np.max([c[0], 1.0 - c[0], c[1], 1.0 - c[1]]) for c in centers])
        least_constrained_idx = np.argmax(weighted_min)
        
        # Apply expansion with constraint validation
        target_total_sum = np.sum(radii) + 0.007
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        expansion_vector = expansion_factor * 1.15 * (1 + 0.05 * np.random.rand(n))
        
        # Create expansion vector with edge-aware adjustment
        new_radii = radii.copy()
        new_radii[least_constrained_idx] = min(new_radii[least_constrained_idx] + expansion_factor * 1.2, 
                                               0.4)  # Max radius constraint
        for i in range(n):
            if i != least_constrained_idx:
                expanded = min(radii[i] + expansion_vector[i], 0.4)
                while expanded > radii[i]:
                    # Apply spatial constraint validation
                    expanded_centers = centers.copy()
                    expanded_centers[2::3] = np.array(new_radii)
                    valid = True
                    for a in range(n):
                        for b in range(a + 1, n):
                            dx = expanded_centers[a, 0] - expanded_centers[b, 0]
                            dy = expanded_centers[a, 1] - expanded_centers[b, 1]
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist < new_radii[a] + new_radii[b] - 1e-12:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        break
                    else:
                        expanded = radii[i] + expansion_vector[i] * 0.8
                new_radii[i] = expanded
        
        # Apply expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate after expansion
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    # Final optimization with spatial-aware tightening
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute final distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Ensure constraint satisfaction
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[i] + radii[j] - 1e-12:
                    # Adjust smaller radius to maintain constraint
                    if radii[i] < radii[j]:
                        radii[i] -= 1e-5
                    else:
                        radii[j] -= 1e-5
        
        v_final = v.copy()
        v_final[2::3] = radii
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())