import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with geometric hashing for better initial distribution
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply randomized geometric hashing to create non-uniform distribution
        x = x_center + np.random.uniform(-0.06, 0.06) * (1 - row * 0.1)
        y = y_center + np.random.uniform(-0.06, 0.06) * (1 - row * 0.1)
        # Shift alternate rows for staggered grid
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
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3*n matches v0

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints using closed-form calculations
    cons = []
    for i in range(n):
        # Left/right boundaries
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Top/bottom boundaries
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing
    def constraint_func(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(v, i, j)})
    
    # Primary optimization with improved configuration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12})
    
    # Post-optimization: controlled expansion on the least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Precompute pairwise distances for efficient selection
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute and apply controlled expansion
        total_sum = np.sum(radii)
        # Calculate a more aggressive expansion factor based on current spacing
        expansion_factor = 0.0085 / (n - 1)  # More aggressive than before
        # Apply expansion to least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector and reoptimize
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    # Additional fine-tuning: random geometric hashing
    if res.success:
        v = res.x
        # Create randomized geometric hash for new configuration
        random_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    
    # Final optimization with improved convergence
    if res.success:
        v = res.x
        radii = v[2::3]
        # Additional random shaking for fine-tuning
        radii = np.clip(radii, 1e-6, None)
        # Randomly perturb positions to avoid flat regions
        perturb_v = v.copy()
        for i in range(n):
            perturb_v[3*i] += np.random.uniform(-0.01, 0.01)
            perturb_v[3*i+1] += np.random.uniform(-0.01, 0.01)
            perturb_v[3*i+2] += np.random.uniform(-0.001, 0.001)
        res = minimize(neg_sum_radii, perturb_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())