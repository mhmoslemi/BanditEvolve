import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Generate initial positions with adaptive clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Use more pronounced spatial perturbation for diversity
        x = x_center + np.random.uniform(-0.1, 0.1)
        y = y_center + np.random.uniform(-0.1, 0.1)
        # Stagger rows for better inter-enclosure
        if row % 2 == 1:
            x += 0.5 / cols * 0.95  # Reduce row shift to minimize crowding
        xs.append(x)
        ys.append(y)
    
    # Use smaller base radius based on grid density for better expansion potential
    r0 = 0.25 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        # Use vectorized operations for efficiency and better convergence
        return -np.sum(v[2::3])

    # Vectorized boundary constraints (inequality: distance >= radius)
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints (non-overlapping requirement)
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with captures to ensure constraint scope
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2})
    
    # Initial optimization with enhanced parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "eps": 1e-9})
    
    # Implement 'shake' heuristic: targeted perturbation of smallest circles
    if res.success:
        v = res.x
        # Calculate current radii
        radii = v[2::3]
        # Find circle with smallest radius that is not at boundary limit
        min_radius_idx = np.argmin(radii[radii > 1e-5])
        
        # Apply controlled spatial perturbation to smallest radius
        perturbation_factor = np.random.uniform(0.05, 0.15)
        # Perturb positions away from neighbors to allow expansion
        dx = np.random.uniform(-0.01, 0.01)
        dy = np.random.uniform(-0.01, 0.01)
        v[3*min_radius_idx] += dx
        v[3*min_radius_idx+1] += dy
        
        # Re-evaluate with perturbed positions
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Final targeted expansion on least constrained circle, with geometric-aware growth
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distance matrix in vectorized way
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate minimum distance to others
        min_dist = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dist)
        
        # Calculate potential expansion factor with dynamic scaling
        expansion_amount = np.min(0.005 * (1.0 + 0.5 * (min_dist[least_constrained_idx] / np.mean(radii)) ** 0.7))
        
        # Apply expansion to all circles (with more to the least constrained)
        for i in range(n):
            if i == least_constrained_idx:
                # Add 25% more expansion to the least constrained
                v[3*i+2] += expansion_amount * 1.25
            else:
                # Spread expansion to others
                v[3*i+2] += expansion_amount * (0.9 + 0.1 * np.random.rand())

        # Re-evaluate with expanded configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())