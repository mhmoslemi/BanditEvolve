import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Initial grid construction with hexagonal lattice structure and optimized perturbation
    # Integrate geometric hashing and dynamic scaling
    
    grid_x = (np.arange(cols) + 0.5) / cols
    grid_y = (np.arange(rows) + 0.5) / rows
    
    # Base radius computation with dynamic spatial scaling and initial perturbation
    # Incorporate adaptive radius scaling based on column density
    base_radius_col = 0.35 / cols
    base_radius_row = 0.34 / rows
    r0 = np.empty(n)
    
    # Initialize with a hybrid grid system
    xs = []
    ys = []
    
    # Generate initial configuration from a geometric tiling pattern
    for i in range(n):
        # Calculate row and column indices
        col = i % cols
        row = i // cols
        # Base positions
        x_base = grid_x[col] + 0.0
        y_base = grid_y[row] + 0.0
        
        # Apply staggered offset
        x = x_base + (0.25 / cols) if row % 2 == 1 else x_base
        y = y_base
        
        # Randomly perturb for diversity
        x += np.random.uniform(-0.05, 0.05) * (1 / np.power(cols, 0.6))
        y += np.random.uniform(-0.05, 0.05) * (1 / np.power(rows, 0.6))
        
        # Ensure position in bounds
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        xs.append(x)
        ys.append(y)
    
    # Initialize base radii with density-aware scaling
    r0 = np.max([base_radius_col, base_radius_row]) * (np.sqrt(n) / cols) - 1e-3
    
    # Define vector and bounds with proper 3*n length
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Ensure bounds are exactly 3n entries
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-4, 0.5))  # radius
    
    # Objective function: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraint setup with geometric hashing and boundary constraints
    cons = []
    
    # Boundary constraints for each circle with tight tolerance (x, y, x-r, y-r)
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Circle-circle non-overlap constraints with geometric hashing (dynamic resolution)
    for i in range(n):
        for j in range(i + 1, n):
            # Weights for non-overlap based on radius distribution
            rad_w = (v0[2::3][i] + v0[2::3][j]) / (np.sum(v0[2::3]) + 1e-12)
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j]) ** 2
                    + (v[3*i+1] - v[3*j+1]) ** 2
                    - (v[3*i+2] + v[3*j+2]) ** 2
                )
            })
    
    # Initial optimization with enhanced parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-11, "gtol": 1e-10})
    
    # Secondary perturbation and spatial reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized spatial constraint analysis
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute spatial constraint metrics
        min_dists = np.min(dists, axis=1)
        min_dists = np.where(min_dists == 0, 1e-12, min_dists)  # Avoid zeros
        total_constrained_idx = np.argmin(min_dists)  # Most constrained circle
        
        # Geometric tiling expansion on most constrained circle
        current_total = np.sum(radii)
        expansion_target = current_total + 0.0075
        expansion_rate = (expansion_target - current_total) / (n - 1) * 1.2
        radius_gain = expansion_rate
        
        # Stochastic expansion with adaptive radius scaling
        new_radii = radii.copy()
        for i in range(n):
            if i == total_constrained_idx:
                new_radii[i] += radius_gain * 1.2
            else:
                # Introduce random expansion with decreasing influence on closer circles
                # Weights decrease with minimum distance
                if min_dists[i] < 0.05:
                    expansion = radius_gain * 0
                elif min_dists[i] < 0.1:
                    expansion = radius_gain * 0.5
                elif min_dists[i] < 0.15:
                    expansion = radius_gain * 0.7
                else:
                    expansion = radius_gain * 0.9 + np.random.rand() * radius_gain * 0.2
                new_radii[i] += expansion
        
        # Create and test expanded configuration
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validation of expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # Adjust expansion by 5% step
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update and reoptimize
        v_new = expanded_v
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-10})
    
    # Final check and fallback if optimization fails
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())