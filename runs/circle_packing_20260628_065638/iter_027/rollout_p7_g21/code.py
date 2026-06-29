import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with lambda closures and parameterized capture
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Shake heuristic: perturb smallest circles and re-optimization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Find smallest radius and calculate their total contribution
        smallest_radius_idx = np.argmin(radii)
        smallest_radius_value = radii[smallest_radius_idx]
        
        # Apply controlled shaking: small perturbation to centers with radius scaling
        shake_strength = 0.002
        shake_radius_weight = 0.3
        shake_center_weight = 0.7
        shake_v = v.copy()
        for i in range(n):
            if i == smallest_radius_idx:
                # Use higher perturbation for smallest radius to "shake out" the local minima
                shake_center = np.random.uniform(-0.05, 0.05, 2)
                shake_radius = 0.001
                shake_v[3*i] += shake_center[0] * (radii[i] / smallest_radius_value) * shake_center_weight
                shake_v[3*i+1] += shake_center[1] * (radii[i] / smallest_radius_value) * shake_center_weight
                shake_v[3*i+2] += shake_radius * (radii[i] / smallest_radius_value) * shake_radius_weight
            else:
                # Slight perturbation for others to encourage reconfiguration
                shake_center = np.random.uniform(-0.005, 0.005, 2)
                shake_v[3*i] += shake_center[0] * (radii[i] / np.mean(radii)) * shake_center_weight
                shake_v[3*i+1] += shake_center[1] * (radii[i] / np.mean(radii)) * shake_center_weight

        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, shake_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Targeted radius expansion with non-overlap check and adaptive expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Efficient vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify circles with minimal constraint (maximized minimum distance)
        min_dist = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dist)
        
        # Targeted expansion with adaptive scaling to maintain feasibility
        target_total_sum = np.sum(radii) + 0.006
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Create expansion vector based on geometric scaling of expansion factor
        expansion = np.full(n, expansion_factor)
        expansion[least_constrained_idx] *= 1.2  # Slightly increase expansion for least constrained
        
        # Apply expansion with constraint validation
        new_radii = radii + expansion
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
                # Reduce expansion if constraint violation detected
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector with final expansion
        v = expanded_v

    # Final adjustment to prevent radius underflow and ensure feasibility
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())