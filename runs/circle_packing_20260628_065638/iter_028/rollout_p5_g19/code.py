import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))  # 6 columns for more efficient spatial hashing
    rows = (n + cols - 1) // cols
    
    # Smart initialization with multi-scale spatial hashing and adaptive clustering
    xs = []
    ys = []
    scale_factors = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]  # Multi-scale spatial distribution
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols * scale_factors[row % len(scale_factors)]
        y_center = (row + 0.5) / rows * scale_factors[col % len(scale_factors)]
        # Apply asymmetric spatial jitter based on local density
        x = x_center + np.random.uniform(-0.06, 0.03) / (0.5 + (col % 3))
        y = y_center + np.random.uniform(-0.03, 0.06) / (0.5 + (row % 3))
        # Row alternation with dynamic shift to prevent gridlock
        if row % 2 == 1 and col % 3 < 2:
            x += 0.4 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.3 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Create bounds with consistent 3n entries
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i (optimized closures)
    
    cons = []
    for i in range(n):
        # LEFT constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # RIGHT constraint: 1.0 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # BOTTOM constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # TOP constraint: 1.0 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using lambda with closure for i,j
    for i in range(n):
        for j in range(i + 1, n):
            # Use closure with fixed i, j for constraint function
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})
    
    # First-stage optimization with tighter convergence and adaptive tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-12})

    # Asymmetric reconfiguration: stochastic spatial perturbation based on spatial density
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Spatial density grid for perturbation focus
        grid_dist = np.zeros(n)
        for i in range(n):
            grid_dist[i] = np.sqrt((centers[i,0] - 0.5)**2 + (centers[i,1] - 0.5)**2)
        max_dist = np.max(grid_dist)
        
        # Generate perturbation map with density-weighted perturbation
        perturbation_factor = np.exp(-grid_dist / max_dist) * 0.05
        spatial_perturbation = np.random.rand(n, 2) * perturbation_factor[:, np.newaxis]
        
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_perturbation[i, 0]
            perturbed_v[3*i+1] += spatial_perturbation[i, 1]
        
        # Re-optimization with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-10})
    
    # Adaptive targeted expansion - identify most flexible circle based on local pressure
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pressure via minimum Euclidean distance from others
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        dist_diagnostics = np.zeros(n)
        for i in range(n):
            dist_diagnostics[i] = np.min(np.max(dists[i, :], initial=1.0))
        
        # Find circle with highest flexibility (least proximity constraints)
        flexible_idx = np.argmax(dist_diagnostics)
        
        # Compute total growth target based on current radius sum and local expansion potential
        current_total = np.sum(radii)
        # Dynamic growth: expand by 0.7% of current total sum
        target_growth = current_total * 0.007
        expansion_amount = target_growth / (n)  # Distribute evenly, but focus first on flexible
        
        # Create expansion vector and apply targeted expansion
        new_radii = radii.copy()
        new_radii[flexible_idx] += expansion_amount * 1.1  # Slight over-provision for reconfiguration
        for i in range(n):
            if i != flexible_idx:
                new_radii[i] += expansion_amount
        
        # Apply expansion in a safe mode with gradient validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Gradient-based validation check instead of brute-force distance
            grad = np.zeros_like(expanded_v)
            for i in range(n):
                # Compute gradient from constraints (only for overlapping circles)
                for j in range(n):
                    if i != j:
                        dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < new_radii[i] + new_radii[j] - 1e-12:
                            # Compute gradient contribution to distance constraint
                            grad[3*i+2] += 2 * (new_radii[i] + new_radii[j]) / dist
                            grad[3*j+2] += 2 * (new_radii[i] + new_radii[j]) / dist
                            # Penalize the gradient to encourage expansion
                            grad[3*i+2] -= 2
                            grad[3*j+2] -= 2
            
            # If no gradients are positive, we are within bounds
            if np.all(grad[2::3] <= 0):
                break
            # Otherwise, reduce expansion slightly
            new_radii = radii + (new_radii - radii) * 0.96
        
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with refined configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "eps": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())