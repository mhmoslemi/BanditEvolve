import numpy as np

def run_packing():
    n = 26
    # Optimal column structure with adaptive grid
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    col_range = 0.95 / cols
    row_range = 0.95 / rows

    # Advanced initializer with dynamic perturbation and hierarchical clustering
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        # Base grid with staggered rows
        base_x = (col + 0.5) * col_range
        base_y = (row + 0.5) * row_range
        # Apply hierarchical clustering with random perturbations
        x = base_x + np.random.uniform(-0.04, 0.04)
        y = base_y + np.random.uniform(-0.04, 0.04)
        # Stagger alternate rows
        if row % 2 == 1:
            x += 0.5 * col_range
        xs.append(x)
        ys.append(y)
    
    # Initial radius with adaptive scaling
    r0 = 0.45 / max(cols, rows) - 1e-3
    r0 = np.clip(r0, 1e-4, 0.5)

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0.copy()

    # Ensure length matches vector
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with lambda captures
    cons = []
    for i in range(n):
        # Left bound: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right bound: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom bound: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top bound: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized pairwise distance constraints using lambda capture
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                                         - (v[3*i+2] + v[3*j+2])**2
            })

    # First optimization with refined settings
    res1 = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                    constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})

    # First shake heuristic: perturb smallest circles with geometric scaling
    if res1.success:
        v = res1.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Shuffle indices to randomize perturbation order
        idx = np.random.permutation(n)
        for i in range(n):
            # Perturb smallest circles more to enable escape
            if i < n//2:
                # Use relative scaling based on radii to maintain spatial integrity
                scale = np.clip(0.1 * (1 + np.sqrt(radii[idx[i]])), 0.05, 0.25)
                px, py = np.random.uniform(-scale, scale, 2)
                v[3*idx[i]] += px
                v[3*idx[i]+1] += py
        
        # Second optimization with refined settings
        res2 = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                        constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Second shake heuristic: perturb mid-sized circles with spatial-aware perturbation
    if res2.success:
        v = res2.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distance to neighbors to guide spatial perturbation
        distances = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Find mid-sized circles with moderate constraint
        mid_radii = radii[np.argsort(radii)[n//2:3*n//4]]
        mid_idx = np.random.choice(np.where(np.isclose(radii, mid_radii, atol=1e-8))[0], size=5, replace=False)
        
        for i in mid_idx:
            # Use relative scaling based on radii and neighborhood distance
            scale = np.clip(np.sqrt(radii[i])/10, 0.03, 0.15)
            px, py = np.random.uniform(-scale, scale, 2)
            v[3*i] += px
            v[3*i+1] += py
        
        # Third optimization with tighter tolerances
        res3 = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                        constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Final optimization with advanced radius balancing
    if res3.success:
        v = res3.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle (one with most free space)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Least constrained circle

        # Calculate growth based on current total sum and potential for expansion
        current_total = np.sum(radii)
        target_growth = 0.006  # Target of 0.006 increase in total sum
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        
        # Apply expansion with constraint validation
        max_iter = 5
        success = False
        for _ in range(max_iter):
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                v = expanded_v
                success = True
                break
            
            # If invalid, decrease expansion slightly
            new_radii = new_radii * 0.99  # Slow down expansion

        # Final optimization with expanded configuration
        if success:
            res4 = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                            constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
        else:
            res4 = res3
    
    v = res4.x if res4.success else res3.x if res3.success else res2.x if res2.success else res1.x
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    sum_radii = float(radii.sum())
    return centers, radii, sum_radii