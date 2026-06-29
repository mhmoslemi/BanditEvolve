import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with a more structured, adaptive grid for better initial conditions
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply more precise randomized offset and row staggering
        x = x_center + np.random.uniform(-0.06, 0.06) * (1 - 0.6 * row)
        y = y_center + np.random.uniform(-0.06, 0.06) * (1 - 0.6 * row)
        # Alternate row staggering
        if row % 2 == 1:
            x += 0.5 / cols * (1 - 0.6 * row)
        xs.append(x)
        ys.append(y)
    
    # Start radii with adaptive scaling to ensure more even distribution
    initial_radii = 0.4 / cols + np.random.uniform(-0.02, 0.02, n)
    initial_radii = np.clip(initial_radii, 1e-4, 0.05)
    r0 = initial_radii.mean()
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0 * np.ones(n)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries (using more explicit lambda definitions)
    cons = []
    for i in range(n):
        # Left side constraint
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right side constraint
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom constraint
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top constraint
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using lambda with i, j capture
    for i in range(n):
        for j in range(i + 1, n):
            # Use a lambda with explicit i and j to prevent closure capture issues
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-10, "eps": 1e-9})

    # Apply iterative 'shake' heuristic on least constrained circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances for non-overlap
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the circle with the largest minimum distance to its neighbors
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Generate perturbations proportional to circle's current radius
        perturbation_scale = np.clip(0.001 * (radii[least_constrained_idx]/radii.mean()), 0.0005, 0.002)
        perturbation = np.random.normal(0, perturbation_scale, size=2)
        
        # Perturb this circle's center, and re-optimize
        perturbed_v = v.copy()
        perturbed_v[3*least_constrained_idx] += perturbation[0]
        perturbed_v[3*least_constrained_idx + 1] += perturbation[1]
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-9})
        
        # Further targeted expansion on least constrained circle
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Recalculate distances for the new positions
            dx_opt = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy_opt = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists_opt = np.sqrt(dx_opt**2 + dy_opt**2)
            
            # Recalculate least constrained circle
            min_dists_opt = np.min(dists_opt, axis=1)
            least_constrained_idx = np.argmax(min_dists_opt)
            
            # Calculate current total sum and target expansion
            current_total = np.sum(radii)
            target_growth = 0.01
            expansion_amount = target_growth / (n - 1) * (current_total / np.sum(radii))
            
            # Create expansion vector
            expansion_vector = np.full(n, expansion_amount)
            expansion_vector[least_constrained_idx] += expansion_amount * 0.2  # slight over-expansion
            
            # Apply expansion with validation
            v_expanded = v.copy()
            v_expanded[2::3] = radii + expansion_vector
            
            # Recalculate distances with expanded radii
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists_expanded = np.sqrt(dx**2 + dy**2)
            
            # Validate expansion
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    if dists_expanded[i,j] < (v_expanded[3*i+2] + v_expanded[3*j+2]) - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-9})
            else:
                # Revert to original configuration
                pass
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())