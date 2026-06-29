import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Base grid parameters and initial perturbations
    grid_x = (np.arange(cols) + 0.5) / cols
    grid_y = (np.arange(rows) + 0.5) / rows
    
    # Initialize centers with staggered grid and spatial hashing for diversity
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x = grid_x[col] + np.random.uniform(-0.05, 0.05)
        y = grid_y[row] + np.random.uniform(-0.05, 0.05)
        # Staggered grid - shift odd rows right, even rows left
        if row % 2 == 1:
            x += 0.5 / cols
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    
    # Dynamic radius scaling based on packing density
    radius_base = 0.37 / cols  # Slightly higher than previous base
    r0 = radius_base - 1e-3
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Ensure bounds are exactly 3n entries
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-4, 0.5))  # Minimum radius to prevent singularities
    
    # Objective: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraints definition: 4 per circle for boundary (left, right, bottom, top)
    # and N*(N-1)/2 for circle-circle overlap
    cons = []
    
    # Vectorized boundary constraints (left, right, bottom, top)
    for i in range(n):
        # Left constraint: x - r >= 0 => x >= r
        # Fun: x - r -> (v[3*i] - v[3*i + 2]) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right constraint: x + r <= 1 => 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
    
    # Vectorized circle-circle overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Constraint: distance^2 - (r_i + r_j)^2 >= 0
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                    - (v[3*i+2] + v[3*j+2])**2
                )
            })
    
    # Initial optimization with tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", 
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-10})
    
    # If no improvement, fallback and reconfigure
    if not res.success:
        print("Initial optimization failed")
        # Redefine v0 with better initial perturbation and grid logic
        xs = []
        ys = []
        for i in range(n):
            col = i % cols
            row = i // cols
            x = grid_x[col] + np.random.uniform(-0.05, 0.05)
            y = grid_y[row] + np.random.uniform(-0.05, 0.05)
            if row % 2 == 1:
                x += 0.5 / cols
            xs.append(np.clip(x, 0.0, 1.0))
            ys.append(np.clip(y, 0.0, 1.0))
        v0 = np.empty(3 * n)
        v0[0::3] = np.array(xs)
        v0[1::3] = np.array(ys)
        v0[2::3] = r0
    
        res = minimize(neg_sum_radii, v0, method="SLSQP", 
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 800, "ftol": 1e-11, "gtol": 1e-10})
    
    # Trigger spatial reconfiguration with asymmetric perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Spatial perturbation with gradient-aware expansion
        spatial_hash = np.random.rand(n, 2) * 0.03
        perturbed_v = v.copy()
        for i in range(n):
            dx_perturb = spatial_hash[i, 0] * (radii[i] / np.mean(radii) * 1.1)
            dy_perturb = spatial_hash[i, 1] * (radii[i] / np.mean(radii) * 1.1)
            perturbed_v[3*i] += dx_perturb
            perturbed_v[3*i+1] += dy_perturb
        
        # Secondary optimization with same constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", 
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-10})
    
    # Final targeted expansion on least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix and min distances calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)  # Min distance to other circles
        least_constrained_idx = np.argmax(min_dists)  # Circle farthest from others
        
        # Calculate expansion factor with soft constraint-aware scaling
        current_total = np.sum(radii)
        expansion_target = current_total + 0.0075  # Slight increase in total radii
        expansion_rate = (expansion_target - current_total) / (n-1)
        
        # Generate expansion vector with stochastic amplification
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_rate * 1.2  # Over-expand slightly
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_rate * (1.0 + 0.1 * np.random.rand())  # Stochastic expansion
        
        # Final validation and constraint satisfaction
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
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
                # If invalid, decrease expansion by 5%
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update v with expanded radii and refine
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", 
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())