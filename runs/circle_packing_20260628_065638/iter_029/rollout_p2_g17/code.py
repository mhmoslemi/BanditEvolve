import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Base grid parameters and initial perturbations with more precise spatial
    # hierarchy optimization and asymmetric seeding
    grid_x = (np.arange(cols) + 0.5) / cols
    grid_y = (np.arange(rows) + 0.5) / rows

    # Initialize centers with randomized grid and geometric-aware seeding
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        # Base grid positions + fine perturbation
        x = grid_x[col] + np.random.uniform(-0.03, 0.03)
        y = grid_y[row] + np.random.uniform(-0.03, 0.03)
        # Apply staggered pattern - odd rows offset
        if row % 2 == 1:
            x += 0.5 / cols
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation with geometric-aware scaling - increased spacing
    # and tighter grid-specific radius base
    radius_base = 0.35 / cols
    r0 = radius_base - 1e-3
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x in [0,1]
        bounds.append((0.0, 1.0))  # y in [0,1]
        bounds.append((1e-4, 0.5))  # r in (0.0001, 0.5]
    
    # Objective function to minimize (negative total radius)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with stricter tolerance and more consistent lambda
    # structure to avoid capturing issues
    cons = []

    # Boundary constraints for each circle
    for i in range(n):
        i_idx = 3*i
        i_idx_r = i_idx + 2
        # Left-bound: x >= r (x - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right-bound: 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom-bound: y >= r (y - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top-bound: 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Circle-circle distance constraint: distance_sq - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            i_idx = 3*i
            j_idx = 3*j
            i_idx_r = i_idx + 2
            j_idx_r = j_idx + 2
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                             - (v[3*i+2] + v[3*j+2])**2})
    
    # Initial optimization with high-precision settings and vectorized constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-11})
    
    # Handle failure with dynamic re-seeding and tighter configuration
    if not res.success:
        print("Initial optimization failed. Re-seeding with denser spatial hashing.")
        # Re-seed with tighter perturbation and better spatial hashing
        xs = []
        ys = []
        for i in range(n):
            col = i % cols
            row = i // cols
            x = grid_x[col] + np.random.uniform(-0.015, 0.015)
            y = grid_y[row] + np.random.uniform(-0.015, 0.015)
            # Staggered rows for better distribution
            if row % 2 == 1:
                x += 0.4 / cols
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            xs.append(x)
            ys.append(y)
        v0 = np.empty(3 * n)
        v0[0::3] = np.array(xs)
        v0[1::3] = np.array(ys)
        v0[2::3] = np.full(n, r0)
        res = minimize(neg_sum_radii, v0, method="SLSQP", 
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-11})
    
    # Apply asymmetric spatial reconfiguration with gradient-aware movement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial perturbation vector with gradient-aware scaling
        spatial_hash = np.random.rand(n, 2) * 0.025
        perturbed_v = v.copy()
        for i in range(n):
            # Scale perturbations with inverse of mean radius to focus on constrained areas
            inv_mean_r = 1.0 / np.mean(radii) if np.mean(radii) > 1e-8 else 1.0
            dx_perturb = spatial_hash[i, 0] * (radii[i] * inv_mean_r * 1.2)
            dy_perturb = spatial_hash[i, 1] * (radii[i] * inv_mean_r * 1.2)
            # Apply perturbations with clamping
            perturbed_v[3*i] += dx_perturb
            perturbed_v[3*i+1] += dy_perturb
            perturbed_v[3*i] = np.clip(perturbed_v[3*i], 0.0, 1.0)
            perturbed_v[3*i+1] = np.clip(perturbed_v[3*i+1], 0.0, 1.0)
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", 
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-11})
    
    # Advanced targeted expansion on most spatially constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix (sparse for efficiency)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle (max min distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        least_constrained_r = radii[least_constrained_idx]
        mean_r = np.mean(radii)
        
        # Compute expansion vector with gradient-aware expansion
        current_total = np.sum(radii)
        target_growth = 0.0085  # aggressive but safe growth
        expansion_target = current_total + target_growth
        expansion_rate = (expansion_target - current_total) / (n - 1)
        
        # Create expanded radii with asymmetric expansion strategy
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_rate * 1.15  # Over-extend slightly
        for i in range(n):
            if i != least_constrained_idx:
                # Apply stochastic expansion with directional bias
                expansion = expansion_rate * (1 + 0.2 * np.random.rand())
                new_radii[i] += expansion * (radii[i] / mean_r)  # Scale based on current distribution
        
        # Validate expanded configuration and iteratively adjust
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate no overlap
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
                break
            else:
                # Adjust by reducing expansion
                new_radii = radii + (new_radii - radii) * 0.95
                # Early exit if too many iterations occur
                if sum(new_radii) < current_total - 0.0005:
                    break
        
        # Apply expansion and refine
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", 
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-11})
    
    # Final check and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())