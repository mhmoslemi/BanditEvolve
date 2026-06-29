import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols  # 5x6 = 30 cells

    # Initialize positions with optimized grid layout
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        # Start from the center of each cell
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset for diversity
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Stagger alternate rows to reduce vertical clustering
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initialize radii to be slightly smaller than the maximum possible
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

    # Vectorized constraint setup
    cons = []
    for i in range(n):
        # Left boundary constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Efficient pairwise distance constraint using vectorized calculation
    def compute_overlap_constraints(x, y, r):
        # Use broadcasting to compute all pairwise distances
        dx = x[None, :] - x[:, None]
        dy = y[None, :] - y[:, None]
        dists = np.sqrt(dx**2 + dy**2)
        return dists - r[None, :] - r[:, None]
    
    # Add overlap constraints for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: 
                         np.sum((v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - 
                                (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with tighter tolerances and more iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Hybrid reconfiguration using geometric hashing and targeted expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Generate geometric hashing for spatial disruption
        random_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        
        # Re-evaluate with updated configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Final targeted radius expansion for the most constrained circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute pairwise distances using vectorized calculation
        dx = centers[:, 0][:, None] - centers[:, 0][None, :]
        dy = centers[:, 1][:, None] - centers[:, 1][None, :]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute minimal distances for each circle
        min_dists = np.min(dists, axis=1)
        
        # Find the circle with the smallest radius and largest minimum distance
        least_constrained_idx = np.argsort(np.column_stack((radii, min_dists)))[-1][0]
        
        # Calculate expansion factor based on margin
        radius = radii[least_constrained_idx]
        min_dist = min_dists[least_constrained_idx]
        max_radius = min_dist - 1e-6  # Ensure non-overlapping
        expansion_factor = (max_radius - radius) / np.sqrt(2)  # Factor of safety
        
        # Apply expansion to the least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with updated radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())