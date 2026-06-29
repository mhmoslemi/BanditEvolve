import numpy as np

def run_packing():
    n = 26
    
    # Optimal grid configuration for 26 circles (4x6 with 2 extra)
    cols = 6
    rows = 5
    
    # Use geometric hashing and spatial perturbation to initialize centers
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = col / cols
        base_y = row / rows
        # Apply structured perturbation to avoid symmetry
        x = base_x + np.random.uniform(-0.08, 0.08)
        y = base_y + np.random.uniform(-0.08, 0.08)
        # Add staggered offset to alternate rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.24 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    # Objective: maximize sum of radii (minimize -sum_radii)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraints: all boundaries and non-overlapping circles
    cons = []
    for i in range(n):
        # 4 boundary constraints per circle (x + r <= 1, x - r >= 0, y + r <= 1, y - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints: all pairs (i,j) must satisfy distance > r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            # Use delayed lambda with fixed i,j to avoid closure capture bugs
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                             - (v[3*i+2] + v[3*j+2])**2})
    
    # First optimization with increased precision
    res = minimize(
        neg_sum_radii, v0, method="SLSQP", bounds=bounds,
        constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-12}
    )
    
    if not res.success:
        # Fallback: reset to initial conditions if primary optimization failed
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
    
    # Asymmetric reconfiguration with adaptive scaling & soft constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling based on circle sizes
        spatial_hash = np.random.rand(n, 2) * 0.08
        scaled_hash = spatial_hash * (radii / np.mean(radii))[:,np.newaxis]
        
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += scaled_hash[i,0]
            perturbed_v[3*i+1] += scaled_hash[i,1]
        
        # Re-evaluate with new configuration
        res = minimize(
            neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
            constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-12}
        )
    
    # Targeted expansion of minimal constraint circle with soft expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute minimum distance to other circles for constraint analysis
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with maximum minimal distance (least constrained)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
                
        # Target growth: increase total radius by 0.01 from current value
        current_sum = np.sum(radii)
        target_sum = current_sum + 0.01
        delta = (target_sum - current_sum) / (n - 1)
        
        # Create expansion vector with targeted expansion to least constrained circle
        expansion = np.full(n, delta)
        expansion[least_constrained_idx] *= 1.15  # Slightly more expansion
        
        # Apply expansion with iterative refinement
        v_expanded = v.copy()
        v_expanded[2::3] = radii + expansion
        
        # Re-optimize while preserving constraints
        res = minimize(
            neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
            constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-12}
        )
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())