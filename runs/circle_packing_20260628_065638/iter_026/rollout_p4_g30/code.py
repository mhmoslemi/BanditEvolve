import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions using geometric hashing and asymmetric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = col / cols
        base_y = row / rows
        base_x += 0.5 / cols
        base_y += 0.5 / rows
        
        # Asymmetric clustering: expand edges for better spread
        edge_factor = 0.5 if col == 0 or col == cols-1 or row ==0 or row == rows-1 else 0.2
        perturb_x = np.random.uniform(-0.02 * edge_factor, 0.02 * edge_factor)
        perturb_y = np.random.uniform(-0.02 * edge_factor, 0.02 * edge_factor)
        
        x = base_x + perturb_x
        y = base_y + perturb_y
        xs.append(x)
        ys.append(y)
    
    # Set initial radii with more aggressive base radius and less tight min constraint
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with tighter tolerances
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing and tighter tolerances
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization with increased max iterations and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "maxls": 200})
    
    # Disruptive geometric transformation: geometric hashing + forced reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Random geometric hash for forced reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "maxls": 150})
    
    # Targeted radius expansion on circle with smallest non-zero radius
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Find the circle with smallest non-zero radius
        non_zero_mask = radii > 1e-6
        valid_indices = np.where(non_zero_mask)[0]
        if len(valid_indices) > 0:
            min_idx = valid_indices[np.argmin(radii[valid_indices])]
        
        # Create a radius expansion vector with soft enforcement
        new_radii = radii.copy()
        expansion_factor = 0.002  # Base expansion amount
        max_attempts = 50
        
        for _ in range(max_attempts):
            # Expand the selected circle's radius first
            new_radii[min_idx] = min(radii[min_idx] + expansion_factor * 1.2, 0.45)
            for i in range(n):
                if i != min_idx:
                    new_radii[i] = min(radii[i] + expansion_factor * (1.0 + 0.1 * np.random.rand()), 0.45)
            
            # Check for valid configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Update decision vector
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "maxls": 150})
                v = res.x
                radii = v[2::3]
                centers = np.column_stack([v[0::3], v[1::3]])
                break
            else:
                # If invalid, backtrack gradually
                new_radii = radii.copy()
                expansion_factor *= 0.9

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())