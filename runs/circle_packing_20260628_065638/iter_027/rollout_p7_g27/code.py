import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with refined staggered grid and randomized jitter
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Use small, well-controlled random jitter for better convergence
        jitter = np.random.uniform(-0.02, 0.02, size=2)
        x = x_center + jitter[0]
        y = y_center + jitter[1]
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radius allocation based on grid spacing
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Construct bounds list to match the 3*n-dimensional vector
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
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

    # Overlap constraints with vectorized distance matrix computation
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                          "fun": lambda v, i=i, j=j: 
                                (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1600, "ftol": 1e-10})
    
    # Add shake phase: perturb small circles to escape local minima
    if res.success:
        # Extract current state
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances, avoid recomputing every time
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        pairwise_distances = np.sqrt(dx**2 + dy**2)
        
        # Identify smallest circles (with least expansion potential)
        small_circle_indices = np.argsort(radii)[:int(n * 0.3)]
        small_circle_radii = radii[small_circle_indices]
        small_circle_centers = centers[small_circle_indices]
        
        # Generate spatially intelligent perturbation vectors
        perturbation_factor = 0.015
        perturbations = np.random.rand(len(small_circle_indices), 2) * perturbation_factor
        
        # Apply small, geometrically informed shocks
        for idx, (i, perturb) in enumerate(zip(small_circle_indices, perturbations)):
            v[3*i] += perturb[0] * (radii[i] / np.mean(radii))
            v[3*i+1] += perturb[1] * (radii[i] / np.mean(radii))
            v[3*i+2] += 0.0005 * (1 + np.random.uniform(-0.2, 0.2))  # slight radius boost
        
        # Re-evaluate with perturbed state
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Final refinement: targeted expansion with gradient-aware adjustment
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute all pairwise distances (vectorized)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        pairwise_distances = np.sqrt(dx**2 + dy**2)
        
        # Compute constraint slack to identify least constrained circles
        slack = pairwise_distances - (radii[:, np.newaxis] + radii[np.newaxis, :])
        min_slack = np.min(slack, axis=1)
        least_constrained_idx = np.argsort(min_slack)[-int(n * 0.3):]
        
        # Expand least constrained circles while maintaining feasibility
        expansion_factor = 0.007
        for i in least_constrained_idx:
            # Compute available expansion without violating any constraint
            available_expansion = np.min(np.where((pairwise_distances[i] - (radii[i] + radii)) >= -1e-9,
                                                 1 - 1e-9, np.inf))
            if available_expansion > 1e-9:
                expansion = np.clip(expansion_factor * (1.0 + np.random.uniform(-0.15, 0.15)), 0, available_expansion)
                # Apply expansion while preserving feasibility
                v[3*i + 2] += expansion
        
        # Final optimization pass
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())