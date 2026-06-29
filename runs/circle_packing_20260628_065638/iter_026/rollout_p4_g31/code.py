import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with geometric hashing and adaptive offset
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + (np.random.rand() - 0.5) * (0.06 + np.random.rand() * 0.1)
        y = y_center + (np.random.rand() - 0.5) * (0.06 + np.random.rand() * 0.1)
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

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with aggressive parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-12, "maxls": 500, "eps": 1e-10})
    
    # Execute randomized geometric hashing and targeted expansion on smallest radius
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash for reconfiguration
        geometric_hash = np.random.rand(n, 3) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += geometric_hash[i, 0]
            perturbed_v[3*i+1] += geometric_hash[i, 1]
            perturbed_v[3*i+2] += geometric_hash[i, 2]
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "maxls": 200})

    # Trigger forced reorganization by identifying and expanding smallest non-overlapping circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify smallest non-overlapping circle using distance-based isolation
        interaction = np.sum(dists, axis=1)
        isolated_idx = np.argmin(interaction)
        
        # Create a constrained expansion vector based on geometric hashing
        hash_multiplier = np.random.rand(n) * 0.1 + 0.05
        expansion = (np.sum(radii) + 0.008 - np.sum(radii)) * hash_multiplier
        
        new_radii = radii.copy()
        max_attempts = 10
        for _ in range(max_attempts):
            new_radii = radii.copy()
            new_radii[isolated_idx] += expansion[isolated_idx] * 1.15
            for i in range(n):
                if i != isolated_idx:
                    new_radii[i] += expansion[i] * (1.0 + 0.05 * np.random.rand())
            
            # Validate new configuration
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx_exp = centers[i,0] - centers[j,0]
                    dy_exp = centers[i,1] - centers[j,1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break

        v_expanded = v.copy()
        v_expanded[2::3] = new_radii
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "maxls": 150})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())