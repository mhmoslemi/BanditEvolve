import numpy as np

def run_packing():
    n = 26
    cols = 6
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
        x = x_center + np.random.uniform(-0.06, 0.04)
        y = y_center + np.random.uniform(-0.08, 0.03)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.36 / cols - 1e-3
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
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
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-10})
    
    # Asymmetric reconfiguration: stochastic spatial shift
    if res.success:
        v = res.x
        # Create asymmetric randomness vector for spatial shift
        asymmetric_shift = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += asymmetric_shift[i, 0]
            perturbed_v[3*i+1] += asymmetric_shift[i, 1]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted expansion on most constrained circle (least surrounded) with controlled expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
        
        # Find most constrained circle by minimal average distance to others
        avg_dists = np.mean(dists, axis=1)
        most_constrained_idx = np.argmin(avg_dists)
        
        # Calculate potential for expansion through spatial margins
        margins = np.array([
            v[3*most_constrained_idx] + radii[most_constrained_idx],
            1.0 - v[3*most_constrained_idx] + radii[most_constrained_idx],
            v[3*most_constrained_idx + 1] + radii[most_constrained_idx],
            1.0 - v[3*most_constrained_idx + 1] + radii[most_constrained_idx]
        ])
        
        # Calculate current expansion potential
        current_expansion = np.min(margins)
        
        # Apply controlled expansion with a factor that depends on margins
        expansion_factor = max(1.0 - current_expansion / 1.0, 0.8)
        target_radius_ratio = 1.1 + 0.05 * np.random.rand()
        
        # Attempt expansion while preserving non-overlap
        for _ in range(10):
            new_radii = radii.copy()
            new_radii[most_constrained_idx] *= target_radius_ratio
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_v[3*i] - expanded_v[3*j]
                    dy = expanded_v[3*i+1] - expanded_v[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                v = expanded_v
                res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 200, "ftol": 1e-11})
                break
    
    # Final optimization with fine-tuning
    if res.success:
        v = res.x
        # Additional refinement with gradient-based tightening
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())