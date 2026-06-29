import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols

    # Geometric tiling with asymmetric spatial constraints
    xs = []
    ys = []
    for i in range(n):
        if cols > 5:
            row = i // cols
            col = i % cols
            # Non-uniform grid with denser vertical columns
            if row % 3 == 1:
                col += 1
            x_center = (col + 0.5) / cols
            y_center = (row + 0.5) / rows
            x = x_center + np.random.uniform(-0.05, 0.05)
            y = y_center + np.random.uniform(-0.05, 0.05)
            if row % 2 == 1:
                x += 0.25 / cols
            xs.append(x)
            ys.append(y)
        else:
            row = i // cols
            col = i % cols
            x_center = (col + 0.5) / cols
            y_center = (row + 0.5) / rows
            x = x_center + np.random.uniform(-0.06, 0.06)
            y = y_center + np.random.uniform(-0.06, 0.06)
            if row % 2 == 1:
                x += 0.5 / cols
            xs.append(x)
            ys.append(y)

    # Base radius allocation: denser grid reduces initial radius
    r0 = 0.32 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]   # length 3*n, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints for all circles
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
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10})

    # Radical reconfiguration via geometric tiling: non-local spatial hash
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with dynamic scaling for enhanced reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            # Apply asymmetric spatial perturbation based on radii and position
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / (radii.mean() + 1e-8)) ** 0.5
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / (radii.mean() + 1e-8)) ** 0.7
        # Reconfigure with spatial hashing and dynamic constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP",
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 800, "ftol": 1e-11, "eps": 1e-8})

    # Targeted radii expansion with dynamic constraints and spatial awareness
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix with optimized broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify circle with least spatial constraint (largest min distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate growth based on current total sum and potential for expansion
        current_total = np.sum(radii)
        target_total = current_total + 0.0125
        expansion_factor = (target_total - current_total) / (n - 1)
        
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())  # Stochastic expansion
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
                new_radii = radii + (new_radii - radii) * 0.96
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-8})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())