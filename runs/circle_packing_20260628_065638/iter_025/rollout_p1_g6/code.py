import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with optimized clustering and directional bias
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add directional bias to prevent clustering
        if row % 2 == 0:
            x = x_center + np.random.uniform(-0.03, 0.03)
            y = y_center + np.random.uniform(-0.07, 0.07)
        else:
            x = x_center + np.random.uniform(-0.07, 0.07)
            y = y_center + np.random.uniform(-0.03, 0.03)
        
        xs.append(x)
        ys.append(y)
    
    # Set initial radius as proportion of grid spacing with bias toward larger radii
    r0 = (0.32 / (rows * cols)) * np.sqrt(n / 26) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints
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

    # Initial optimization with high precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-9})

    # Geometric hashing for enhanced spatial diversity
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create directional geometric hashing
        hashing_angles = np.random.rand(n) * 2 * np.pi
        hashing_vectors = np.array([np.cos(hashing_angles), np.sin(hashing_angles)]).T * 0.01

        # Apply spatial transformation using geometric hashing
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += hashing_vectors[i, 0]
            new_v[3*i+1] += hashing_vectors[i, 1]
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Identify weakest circle with spatial and radial constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix without explicit loops
        x_coords = centers[:, 0][:, np.newaxis]
        y_coords = centers[:, 1][:, np.newaxis]
        dx = x_coords - x_coords.T
        dy = y_coords - y_coords.T
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with minimal interaction potential
        min_dists = np.min(dists, axis=1)
        interaction_strength = min_dists / (radii + 1e-12)
        weakest_idx = np.argmin(interaction_strength)
        
        # Targeted expansion of the weakest circle with controlled radial redistribution
        target_total = np.sum(radii) + 0.01  # 0.38% increase
        expansion_factor = (target_total - np.sum(radii)) / (n - 1)
        
        # Radial redistribution with spatial stability
        new_radii = radii.copy()
        new_radii[weakest_idx] += expansion_factor * 1.15  # Slight over-expansion
        for i in range(n):
            if i != weakest_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.03 * np.random.rand())
        
        # Validate configuration before optimization
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
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
                # Reduce expansion if necessary
                new_radii = radii + (new_radii - radii) * 0.98
        
        # Final optimization with refined configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())