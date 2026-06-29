import numpy as np

def run_packing():
    n = 26
    cols = 6  # Adjusted from 5 to 6 for better density and balance
    rows = (n + cols - 1) // cols  # Auto compute rows to match number of circles
    
    # Initialize initial positions using more refined geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add asymmetric perturbation that scales with distance to corners
        x = x_center + np.random.uniform(-0.06, 0.06) * (1 - abs(col / cols - 0.5))
        y = y_center + np.random.uniform(-0.06, 0.06) * (1.0 - abs(row / rows - 0.5))
        # Add staggered grid logic with row-dependent offset
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 - abs(col / cols - 0.5))  # Variable stagger by position
        xs.append(x)
        ys.append(y)
    
    # Initial radii estimation using circle packing density heuristic
    # For circular packing in a square, area-based estimation with adjusted spacing
    area_est = (n * 0.5 ** 2) / (n * (np.pi / 4))  # Area of square / density
    r0 = (area_est ** (1 / 2)) / np.sqrt(1 + np.pi / 4)  # Adjust for spacing
    r0 = np.clip(r0, 1e-4, 0.5)  # Clamp between minimal and maximal
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds: 3*n entries for centers and radii
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with proper capturing
    cons = []
    for i in range(n):
        # Left wall
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right wall
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom wall
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top wall
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using lambda with proper capturing
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # Stage 1: Initial optimization with tight tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", 
                   bounds=bounds, constraints=cons, 
                   options={"maxiter": 1000, "ftol": 1e-12})
    
    # Stochastic reconfiguration: Apply dynamic perturbation based on circle status
    # Check for success before proceeding to avoid unnecessary computations
    if res.success:
        v = res.x
        radii = v[2::3]
        
        # Adaptive spatial perturbation based on radii and position
        spatial_hash = np.random.rand(n, 2) * 0.08
        for i in range(n):
            # Scale perturbation by radius for better convergence
            scale = radii[i] / np.mean(radii) ** 0.5
            v[3*i] += spatial_hash[i, 0] * scale
            v[3*i+1] += spatial_hash[i, 1] * scale
        
        # Stage 2: Re-optimization with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-11, "disp": False})
    
    # Stage 3: Post-optimization targeted expansion using gradient-based heuristics
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for all circles
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate minimal distances to neighbors
        min_dists = np.min(dists, axis=1)
        
        # Find circle with most margin for expansion (greatest safety margin)
        margin_indices = 1 - (min_dists / (radii + np.min(radii)))  # Normalize to 0-1
        least_constrained_idx = np.argmax(margin_indices)
        
        # Calculate baseline growth potential
        current_total = np.sum(radii)
        base_growth = 0.005  # Conservative growth estimate
        
        # Targeted expansion with gradient-guided optimization
        # Use local expansion and apply small, incremental changes to avoid overshoot
        # Calculate potential for expansion in a controlled way
        new_radii = radii.copy()
        # Apply expansion to least constrained circle with a safety multiplier
        expansion_base = base_growth * (1.0 + 0.25 * np.random.rand())  # Add variance
        new_radii[least_constrained_idx] += expansion_base
        
        # Distribute some residual growth among other circles with gradient guidance
        residual = base_growth * 0.3
        for i in range(n):
            if i != least_constrained_idx:
                # Use inverse of margin as guide to allocate expansion
                # Larger margins mean more "free space" to expand
                new_radii[i] += residual * (1.0 - margin_indices[i]) / 0.8
        
        # Update decision vector with new radii and refine
        new_v = v.copy()
        new_v[2::3] = new_radii
        
        # Stage 4: Fine-tuning via constrained optimization
        res = minimize(neg_sum_radii, new_v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-12})

    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())