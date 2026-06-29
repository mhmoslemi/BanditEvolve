import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with geometric hashing and staggered clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid with perturbation to enable geometric hashing
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Stagger rows to create offset grid
        if row % 2 == 1:
            base_x += 0.35 / cols
        # Add geometric hashing perturbation
        hash_perturb = np.random.rand(2) * 0.06 - 0.03
        x = base_x + hash_perturb[0]
        y = base_y + hash_perturb[1]
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

    # Vectorized boundary constraints with tighter tolerances
    cons = []
    for i in range(n):
        # Left constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2] + 1e-12})
        # Right constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2] + 1e-12})
        # Bottom constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2] + 1e-12})
        # Top constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2] + 1e-12})
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First round optimization with tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-10})
    
    # Radical reconfiguration: geometric hashing of radii and positions
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric hash map for radius reconfiguration
        hash_map = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_map[i, 0]
            perturbed_v[3*i+1] += hash_map[i, 1]
            perturbed_v[3*i+2] = max(1e-4, min(0.5, radii[i] + np.random.uniform(-0.002, 0.002)))
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-10})
    
    # Targeted reconfiguration of least constrained circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix for constraint analysis
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2) + 1e-12
        
        # Identify least constrained circle by minimal constraint tightness
        constraint_tightness = np.sum(dists, axis=1) - np.sum(radii) + np.sum(np.minimum(dists - radii, 0))
        least_constrained_idx = np.argmin(constraint_tightness)
        
        # Create expansion vector with spatial hashing
        expansion_base = 0.006 / (n - 1) * 1.1
        new_radii = radii.copy()
        
        # Expand least constrained circle with directional perturbation
        for i in range(n):
            if i != least_constrained_idx:
                # Directional expansion based on position
                dir_dx = v[3*i] - v[3*least_constrained_idx]
                dir_dy = v[3*i+1] - v[3*least_constrained_idx+1]
                dist = np.hypot(dir_dx, dir_dy)
                if dist > 1e-6:
                    dir_dx /= dist
                    dir_dy /= dist
                    new_radii[i] += expansion_base * (0.5 + 0.5 * np.random.rand()) * 1.5
            else:
                # Slight over-expansion to force reconfiguration
                new_radii[i] += expansion_base * 1.3
        
        # Apply expansion with constraint validation
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Validate configuration before final optimization
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = expanded_v[3*i] - expanded_v[3*j]
                dy = expanded_v[3*i+1] - expanded_v[3*j+1]
                dist = np.hypot(dx, dy)
                if dist < new_radii[i] + new_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-10})
    
    # Final refinement with spatial hashing and radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply final geometric hashing for fine-tuning
        hash_perturb = np.random.rand(n, 2) * 0.01 - 0.005
        for i in range(n):
            v[3*i] += hash_perturb[i, 0]
            v[3*i+1] += hash_perturb[i, 1]
            # Add small radius adjustment for stability
            v[3*i+2] += np.random.uniform(-0.001, 0.001)
        
        # Final optimization
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())