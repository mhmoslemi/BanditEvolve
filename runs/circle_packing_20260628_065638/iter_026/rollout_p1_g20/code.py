import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Improved initialization with better geometric diversity
    xs = []
    ys = []
    spatial_offsets = np.random.uniform(-0.04, 0.04, n)
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add staggered offset to promote asymmetric spacing
        x = x_center + spatial_offsets[i]
        y = y_center + np.random.uniform(-0.04, 0.04) * (1 if row % 2 == 0 else -1)
        # Add jitter to escape local minima
        x += np.random.uniform(-0.005, 0.005)
        y += np.random.uniform(-0.005, 0.005)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.33 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Better constraint handling with vectorization and closure fixing
    cons = []
    for i in range(n):
        # Left constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with efficient closure
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # First-stage optimization with tighter constraints and better tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-12})

    # Introduce asymmetric spatial configuration to promote new packing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply asymmetric spatial perturbation
        spatial_perturbation = np.random.uniform(-0.02, 0.02, n)
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += spatial_perturbation[i]
            new_v[3*i+1] += spatial_perturbation[i] * (1 if i % 3 == 0 else -1)
        
        # Second-stage optimization
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-12})

    # Identify and expand most under-constrained circle to unlock potential
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle (minimizes min distances)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion to increase total radius sum
        target_total_sum = np.sum(radii) + 0.008
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Apply expansion with directional enforcement
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Overexpand slightly
        for i in range(n):
            if i != least_constrained_idx:
                # Increase with stochastic variation to maintain spatial diversity
                new_radii[i] += expansion_factor * (1.0 + np.random.uniform(-0.2, 0.3))
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate configuration
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
                # Linear decrease for stability
                new_radii = radii + (new_radii - radii) * 0.96
        
        # Final optimization with tighter constraints
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())