import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized, staggered grid and spatial jitter
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Spatial jitter for diversity and escape from symmetry
        x_offset = np.random.uniform(-0.04, 0.04)
        y_offset = np.random.uniform(-0.04, 0.04)
        # Staggered layout
        if row % 2 == 1:
            x_offset += 0.25 / cols
        x = x_center + x_offset
        y = y_center + y_offset
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on grid spacing and margin
    r0 = 0.35 / cols - 1e-4
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Define consistent bounds for all 3n parameters
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Create vectorized constraints with closure capturing i
    cons = []
    for i in range(n):
        # Left wall constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right wall constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom wall constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top wall constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Create distance-based constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})
    
    # First optimization phase: find feasible point
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # If successful, trigger radical reconfiguration
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Apply spatial hashing to trigger non-local reconfiguration
        hash_mask = np.random.rand(n, 2) * 0.05
        
        # Create a new spatial configuration with reordering
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += hash_mask[i, 0]
            new_v[3*i+1] += hash_mask[i, 1]
        
        # Re-optimization with new configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})
        
        # Ensure success before proceeding
        v = res.x if res.success else v
    
    # Now apply targeted radius expansion on circle with least constraint
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute distance matrix using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with the largest minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion potential with total sum constraint
        target_total = np.sum(radii) + 0.008
        expansion_factor = (target_total - np.sum(radii)) / n
        
        # Apply expansion to all circles with bias toward the least constrained one
        expanded_radii = radii + expansion_factor * (1.0 + 0.1 * np.random.rand(n))
        expanded_radii[least_constrained_idx] += expansion_factor * 1.5
        
        # Apply expansion with gradient-based constraints
        v_expanded = v.copy()
        v_expanded[2::3] = expanded_radii
        
        # Re-optimization with new radii
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
        
        # Apply post-expansion refinement
        v = res.x if res.success else v
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())