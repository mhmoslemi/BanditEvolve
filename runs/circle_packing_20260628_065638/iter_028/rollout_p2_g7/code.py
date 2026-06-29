import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize positions with enhanced randomized grid with tighter spatial hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # First, apply deterministic base position
        x = x_center + 0.02 * np.sin(2 * np.pi * i / 26)
        y = y_center + 0.02 * np.cos(2 * np.pi * i / 26)
        # Apply tighter randomized perturbation with anti-alignment
        x += np.random.uniform(-0.025, 0.025)
        y += np.random.uniform(-0.025, 0.025)
        # Stagger rows to avoid symmetry
        if row % 2 == 1:
            x += 0.1 * np.sin(2 * np.pi * i / cols)
            x = np.clip(x, 1e-6, 1 - 1e-6)
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation with adaptive scaling and geometric awareness
    base_radius = 0.36 / cols - 1e-3
    r0 = base_radius * np.ones(n) + np.random.uniform(-1e-3, 1e-3, n)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.36)]  # 3*n entries, tighter radius range

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with lambda with captured i
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with lambda and bounds protection
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j: 
                        np.clip((v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2, -1e6, np.inf))
            })
    
    # Initial optimization with high precision and enhanced iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-12, "eps": 1e-8})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Asymmetric reconfiguration: geometric hashing with adaptive spatial perturbation
        spatial_hash = np.random.rand(n, 2) * 0.04  # tighter spatial randomness
        v_perturb = v.copy()
        for i in range(n):
            v_perturb[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * 1.1
            v_perturb[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * 1.1
            v_perturb[3*i] = np.clip(v_perturb[3*i], 1e-6, 1 - 1e-6)
            v_perturb[3*i+1] = np.clip(v_perturb[3*i+1], 1e-6, 1 - 1e-6)
        
        # Reoptimize with perturbed configuration
        res = minimize(neg_sum_radii, v_perturb, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Targeted radius expansion with soft constraint enforcement and enhanced validation
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)  # Find circle with most space
        
        current_total = np.sum(radii)
        # Target growth: dynamically adjust based on current configuration
        target_growth = 0.01 * (1 + current_total / np.mean(radii))
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Soft constraint-based expansion vector
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Boost on least constrained
        for i in range(n):
            if i != least_constrained_idx:
                # Use adaptive radius-dependent expansion scaling
                new_radii[i] += expansion_factor * (1.0 + np.clip(np.random.rand(), 0, 0.1))
        
        # Apply expansion while checking for violations
        max_attempts = 5
        expand_attempts = 0
        while expand_attempts < max_attempts:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Use vectorized distance computation for validation
            dx_exp = expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0]
            dy_exp = expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1]
            dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
            
            # Validate all pairwise distances
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    if dist_exp[i, j] < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Gradual scaling back if over-expansion
                new_radii = radii + (new_radii - radii) * 0.95
                expand_attempts += 1
        
        # Final refinement with expanded configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        v_new[0::3] = np.clip(v_new[0::3], 1e-6, 1 - 1e-6)
        v_new[1::3] = np.clip(v_new[1::3], 1e-6, 1 - 1e-6)
        
        # Fine-tuning with new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-13, "eps": 1e-9})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.36)
    return centers, radii, float(radii.sum())