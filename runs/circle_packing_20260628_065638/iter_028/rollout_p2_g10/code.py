import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initial perturbation: asymmetric geometric grid with stochastic bias
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = col / cols + 0.5 / cols
        base_y = row / rows + 0.5 / rows
        x = base_x + np.random.uniform(-0.08, 0.08) * (0.5 + np.random.rand())
        y = base_y + np.random.uniform(-0.08, 0.08) * (0.5 + np.random.rand())
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Radius initialization: adaptive and spatially-aware with tighter scaling
    base_radius = 0.4 / cols
    r0 = base_radius * np.ones(n)
    r0 += np.random.uniform(-0.02, 0.02, size=n)
    r0 = np.clip(r0, 1e-4, 0.5)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Ensure bounds list has correct length for 3*n variables
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries

    def neg_sum_radii(v):
        # Use vectorized distance calculations inside objective function for gradient stability
        return -np.sum(v[2::3])
    
    # Vectorized constraint functions using lambda with static i capture
    cons = []
    for i in range(n):
        # Left side constraint: x_i + r_i <= 1.0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right side constraint: x_i - r_i >= 0.0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom side constraint: y_i + r_i <= 1.0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top side constraint: y_i - r_i >= 0.0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with lambda closure
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})
    
    # Initial optimization with aggressive iteration and tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-9})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # First reconfiguration with spatial randomness bias
        # Add spatial hashing with adaptive scale
        spatial_hash = np.random.rand(n, 2) * (0.3 + 0.2 * np.random.rand(n))
        # Perturb centers asymmetrically based on radius
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * 0.3
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * 0.3
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Spatial-aware expansion: find most isolated circle with vectorized distance evaluation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify most isolated circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Create expansion vector with targeted growth
        current_total = np.sum(radii)
        target_growth = 0.0085  # 0.85% increase in sum
        expansion_factor_base = target_growth / (n - 1) * (current_total / np.sum(radii))
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor_base * 1.3  # Slightly over-expanding
        for i in range(n):
            if i != least_constrained_idx:
                expansion_factor = expansion_factor_base * (1.0 + 0.1 * np.random.rand())
                new_radii[i] += expansion_factor
        
        # Apply adaptive validation and constraint refinement
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate pairwise distances with early exit
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
                # Reduce expansion by 6% if invalid
                new_radii = radii + (new_radii - radii) * 0.94
        
        # Final re-evaluation of expanded configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())