import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with optimized staggered grid + perturbations and radial gradient
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Perturb with non-uniform, directional jitter
        delta_x = np.random.normal(0, 0.03) * (0.2 + 0.3 * (row % 2))
        delta_y = np.random.normal(0, 0.03) * (0.2 + 0.3 * (col % 2))
        x = x_center + delta_x
        y = y_center + delta_y
        # Add spatial correlation to alternate rows
        if row % 2 == 1:
            x += 0.5 * delta_x / np.sqrt(1 + delta_x**2)
        xs.append(x)
        ys.append(y)

    r0 = 0.42 / cols - 1e-3  # Initial radius increase from prior to 0.35
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Construct bounds with adaptive radii constraints for better scaling
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        """Objective function to maximize sum of radii"""
        return -np.sum(v[2::3])

    # Vectorized constraint construction with closure capturing (i,j) properly
    cons = []
    for i in range(n):
        # Left/right bounds
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Top/bottom bounds
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Build vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Add constraint with capture of (i,j)
            cons.append({
                "type": "ineq", 
                "fun": lambda v, i=i, j=j: 
                (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            })

    # Initial optimization with high precision and robust parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 1800, 
                       "ftol": 1e-11,  # Tighter tolerance
                       "eps": 1e-12, 
                       "disp": False
                   })
    
    # Force geometric dissection: isolate top two interacting circles
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute pairwise distances for interaction analysis
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        
        # Identify top two interacting pairs
        pair_distances = dists[np.triu_indices(n, k=1)]
        # Weights: 1/r + 1/r^2 for interaction importance
        pair_weights = 1.0 / pair_distances + 1.0 / (pair_distances ** 2)
        pair_indices = np.argsort(pair_weights)[::-1]
        top_pair_idcs = pair_indices[:2]
        
        # Extract the top two interacting circles
        idxs = set()
        for idx in top_pair_idcs:
            i, j = np.unravel_index(idx, (n, n))
            idxs.add(i)
            idxs.add(j)
        interacting_ids = sorted(list(idxs))
        other_ids = [i for i in range(n) if i not in idxs]
        
        # Reconfigure the interacting circles with topological constraint
        # Create a new geometric map that emphasizes separation
        # Create a new perturbation vector
        v_perturbed = v.copy()
        
        # Define a new radial function for the interacting pair
        # We will apply a radial constraint where the two circles must be at least 0.5r apart
        # This forces topological change
        
        # Define new spatial map for the interacting pair with increased spacing
        # We'll apply a non-uniform scaling to the interacting circles
        perturb_factor = 0.1 + 0.3 * (np.random.rand() ** 2)
        # For the interacting circles, shift horizontally and vertically
        for i in interacting_ids:
            v_perturbed[3*i] += 0.05 * np.random.randn() * (0.2 if i in interacting_ids else 0)
            v_perturbed[3*i+1] += 0.05 * np.random.randn() * (0.2 if i in interacting_ids else 0)
        
        # Re-evaluate the configuration
        res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 600, 
                           "ftol": 1e-11, 
                           "eps": 1e-12, 
                           "disp": False
                       })

    # Targeted radius expansion on the least constrained circle 
    # Use vectorized evaluation for efficiency and safety 
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        
        # Create minimal distance vector to find the least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute radial gradients and spatial gradients
        # We will implement a controlled expansion with gradient feedback
        # First, compute current radii
        current_total = np.sum(radii)
        radius_growths = 1.0 + 0.5 * (np.random.rand(n) - 0.5)  # Stochastic perturbation
        
        # Add a new constraint to the system to enforce a specific reordering
        # Add a constraint that forces the two most interacting circles to be at least 1.5x their sum apart
        # This triggers a global topology shift
        
        # Create a new constraint: 
        # (x1 - x2)^2 + (y1 - y2)^2 >= (r1 + r2)**2 * (1.5)
        # This forces a major reconfiguration
        # Select the top two interacting circles again for this constraint
        pair_distances = dists[np.triu_indices(n, k=1)]
        pair_weights = pair_distances
        idxs = np.argsort(pair_weights)
        top_pair_idx = idxs[-1]
        i, j = np.unravel_index(top_pair_idx, (n, n))
        new_cons = {
            "type": "ineq",
            "fun": lambda v, i=i, j=j: 
            (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2 * 2.25
        }
        cons.append(new_cons)
        
        # Re-evaluate with new topological constraint
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 800, 
                           "ftol": 1e-11, 
                           "eps": 1e-12, 
                           "disp": False
                       })

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    
    # Final cleanup and validation
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        # Apply bounds fixing for edge cases
        if x - r < -1e-12 or x + r > 1 + 1e-12:
            v[3*i] = max(min(x, 1.0), 0.0)
        if y - r < -1e-12 or y + r > 1 + 1e-12:
            v[3*i+1] = max(min(y, 1.0), 0.0)
        v[3*i+2] = np.clip(v[3*i+2], 1e-6, 0.5)
    
    # Final optimization pass with tight tolerances
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 200, 
                       "ftol": 1e-12, 
                       "eps": 1e-12, 
                       "disp": False
                   })
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())