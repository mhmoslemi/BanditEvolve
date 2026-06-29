import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add spatial perturbation to avoid symmetry
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Staggering for staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        if np.random.rand() < 0.3:
            # Add row-wise jitter
            y += np.random.uniform(-0.05, 0.05)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.45 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries exactly

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create constraints with capture of current i in lambda
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Add overlapping constraint with vectorization and lambda capture
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2)
            })

    # Primary optimization with hybrid configuration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-11})
    
    # Implement forced geometric dissection on two most interacting circles
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        interaction = np.sum(dists, axis=1)
        top_idx = np.argsort(interaction)[-2:]  # Two most interacting circles
        
        # Reconfigure the top two circles with forced radius growth and spatial adjustment
        # Initialize with a slight displacement
        new_v = v.copy()
        for i in top_idx:
            new_v[3*i] += np.random.uniform(-0.06, 0.06)
            new_v[3*i+1] += np.random.uniform(-0.06, 0.06)
            new_v[3*i+2] += np.random.uniform(0.002, 0.005)  # Initial radius expansion
        
        # Apply targeted spatial repositioning with gradient-aware perturbation
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
        
        # Final radius expansion of least constrained circle with adjacency constraint
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            
            # Compute distance to all other circles
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Find circle with highest minimum distance (least constrained)
            min_dists = np.min(dists, axis=1)
            isolated_idx = np.argmax(min_dists)
            
            # Ensure isolated circle has space for expansion
            current_total = np.sum(radii)
            target_growth = 0.007
            expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
            
            # Create expansion vector with enhanced expansion for isolated
            expanded_radii = radii.copy()
            expanded_radii[isolated_idx] += expansion_factor * 1.3  # Over-expand slightly
            for i in range(n):
                if i != isolated_idx:
                    # Apply expansion with slight stochastic adjustment
                    expanded_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())
            
            # Re-evaluate with expanded radii and maintain configuration
            while True:
                expanded_v = v.copy()
                expanded_v[2::3] = expanded_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                
                # Validate expanded configuration
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx_ = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy_ = expanded_centers[i, 1] - expanded_centers[j, 1]
                        dist = np.sqrt(dx_**2 + dy_**2)
                        if dist < expanded_radii[i] + expanded_radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
                else:
                    # If invalid, scale down radius expansion with exponential decay
                    expanded_radii = radii + (expanded_radii - radii) * 0.95

            # Update decision vector with new radii
            v = expanded_v
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())