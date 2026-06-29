import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using Poisson disk sampling for even distribution
    # We'll use a simplified geometric pattern for initialization to avoid grid bias
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Generate base position on a regular grid to seed the Poisson-like pattern
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        # Apply jitter to break symmetry and simulate Poisson distribution
        x_jitter = np.random.uniform(-0.13, 0.13)
        y_jitter = np.random.uniform(-0.13, 0.13)
        
        # Apply alternating row shifting for stagger
        if row % 2 == 1:
            x_base += 0.5 / cols * 1.1  # Increased stagger to avoid grid lines
        x = x_base + x_jitter
        y = y_base + y_jitter
        
        # Limit positions to square boundaries
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        xs.append(x)
        ys.append(y)
    
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

    # Vectorized constraints for boundaries using lambda with captured i
    cons = []
    for i in range(n):
        # Left boundary constraint: x - r >= 0
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary constraint: x + r <= 1
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary constraint: y - r >= 0
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary constraint: y + r <= 1
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Use more stable gradient-based optimization to handle constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-12})
    
    # Analyze the solution
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for all circles for constraint analysis
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute the minimum distance to others for each circle
        min_dist_per_circle = np.min(dists[dists > 1e-8], axis=1)
        # Identify circle with the smallest minimum distance
        most_constrained_idx = np.argmin(min_dist_per_circle)
        
        # Define the most constrained circle as the focus
        if min_dist_per_circle[most_constrained_idx] > 0:
            # Compute the sum of all current radii
            current_total = np.sum(radii)
            target_growth = 0.006
            expansion_factor_global = target_growth / (n - 1) * (current_total / np.sum(radii) ** 0.7)
            
            # Create a vector of proposed radii with expansion for the most constrained
            expanded_radii = radii.copy()
            expanded_radii[most_constrained_idx] += expansion_factor_global * 1.15
            for i in range(n):
                if i != most_constrained_idx:
                    expanded_radii[i] += expansion_factor_global * (1.0 + np.random.uniform(0.0, 0.15))
            
            # Validate the expanded configuration
            valid = True
            new_dists = np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 + 
                               (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2)
            for i in range(n):
                for j in range(i+1, n):
                    if new_dists[i,j] < expanded_radii[i] + expanded_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Use the expanded radii with a soft global constraint
                res = minimize(neg_sum_radii, np.concatenate([v[0::3], v[1::3], expanded_radii]), 
                              method="SLSQP", bounds=bounds,
                              constraints=cons, 
                              options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-12})
        
                v = res.x if res.success else v
    
    # Final validation and extraction
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())