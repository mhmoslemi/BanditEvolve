import numpy as np

def run_packing():
    n = 26
    cols = 6  # Increase grid columns to 6 for balanced layout
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add small randomized offset to break symmetry
        x = x_center + np.random.uniform(-0.07, 0.07)
        y = y_center + np.random.uniform(-0.07, 0.07)
        # Shift alternate rows for staggered grid
        if row % 2 == 1:
            x += 0.55 / cols * (1.0 if row % 4 == 1 else -1.0)
        xs.append(x)
        ys.append(y)
    
    # Use a more refined initial radius based on grid spacing and padding
    r0 = 0.34 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Ensure bounds list is consistent with 3*n parameters
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Boundary constraints with careful closure handling
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
        
    # Overlap constraints with vectorized and stable closure handling
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization with high precision and convergence control
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-10, "eps": 1e-12, "disp": False})
    
    # Asymmetric reconfiguration with spatial hashing and adaptive perturbation
    if res.success:
        v = res.x
        # Create dynamic spatial hash for asymmetric perturbation
        spatial_hash = np.random.rand(n, 2) * 0.05
        # Apply spatial hashing with adaptive scaling
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (1.0 + np.random.rand() * 0.3)
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (1.0 + np.random.rand() * 0.3)
        
        # Re-evaluate with stochastic spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-11, "disp": False})
    
    # Targeted radius expansion with isolation-aware optimization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Efficient vectorized distance matrix using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by minimizing min distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Use a constrained expansion to unlock potential
        current_total = np.sum(radii)
        target_total = current_total + 0.008  # Increase expansion amount
            
        # Apply controlled expansion to isolated circle with soft constraints
        expansion_factor = (target_total - current_total) / (n - 1)
        new_radii = radii.copy()
        expanded_v = v.copy()
        
        # Expand isolated circle with higher priority
        new_radii[least_constrained_idx] = min(
            np.clip(radii[least_constrained_idx] + expansion_factor * 1.3, 1e-4, 0.5), 
            0.5 - max(dists[least_constrained_idx]) * 0.85)
        
        # Expand others with moderate expansion and spatial constraints
        for i in range(n):
            if i != least_constrained_idx:
                # Apply soft expansion with spatial constraint enforcement
                max_allowed = max(1e-4, 0.5 - max(dists[i]) * 0.85)
                new_radii[i] = np.clip(radii[i] + expansion_factor * 0.8, 1e-4, max_allowed)
        
        # Apply expansion and validate with constraint checking
        expanded_v[2::3] = new_radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-11, "disp": False})
    
    # Final validation and cleanup
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())