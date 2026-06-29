import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a geometric hashing scheme with staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply geometric hashing for spatial diversity
        x_hash = np.random.normal(0, 0.06, size=1)
        y_hash = np.random.normal(0, 0.06, size=1)
        # Shift alternate rows for staggered grid
        if row % 2 == 1:
            x_hash += 0.04 / cols * np.random.choice([-1, 1])
        x = x_center + x_hash
        y = y_center + y_hash
        xs.append(x)
        ys.append(y)
    
    r0 = 0.32 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with tighter tolerances
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: (1.0 - v[3*i] - v[3*i+2]), "jac": lambda v, i=i: (-1.0, 0.0, -1.0)})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: (v[3*i] - v[3*i+2]), "jac": lambda v, i=i: (1.0, 0.0, -1.0)})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: (1.0 - v[3*i+1] - v[3*i+2]), "jac": lambda v, i=i: (0.0, -1.0, -1.0)})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: (v[3*i+1] - v[3*i+2]), "jac": lambda v, i=i: (0.0, 1.0, -1.0)})

    # Vectorized overlap constraints with geometric hashing and gradient estimation
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            # Gradient estimation for constraint function
            def jac_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                grad_dx = 2 * dx / dist
                grad_dy = 2 * dy / dist
                grad_r1 = -2 * dist
                grad_r2 = -2 * dist
                return (
                    grad_dx,  # dx derivative
                    grad_dy,  # dy derivative
                    grad_r1,  # r1 derivative
                    grad_r2   # r2 derivative
                )
            cons.append({"type": "ineq", "fun": constraint_func, "jac": jac_func})

    # Initial optimization with increased precision and robustness
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "maxls": 200})

    # Geometric hashing reconfiguration with spatial perturbation and gradient smoothing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric hash for spatial perturbation
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with perturbed spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "maxls": 100})

    # Targeted geometric reordering with spatial constraint prioritization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute spatial interaction matrix
        interaction = np.sum(dists, axis=1)
        isolated_idx = np.argmax(interaction)

        # Apply multi-stage radius expansion with spatial constraint enforcement
        while True:
            # Calculate total current sum
            total_sum = np.sum(radii)
            # Define expansion goal: incremental 0.007 increase
            target_total_sum = total_sum + 0.007
            
            # Create expansion vector
            expansion = (target_total_sum - total_sum) / (n - 1)
            
            # Create new radii with soft expansion and spatial constraints
            new_radii = radii.copy()
            new_radii[isolated_idx] += expansion * 1.2  # Over-expansion for reconfiguration
            
            # Add randomized expansion to other circles
            for i in range(n):
                if i != isolated_idx:
                    new_radii[i] += expansion * (1.0 + 0.1 * np.random.rand())
            
            # Validate with constraint checking
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If constraints are violated, reduce expansion by 5%
                new_radii = radii + (new_radii - radii) * 0.95

        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "maxls": 100})

    # Final optimization with geometric hashing and gradient estimation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Final validation with gradient estimation
        total_sum = np.sum(radii)
        while True:
            # Recalculate distances with current configuration
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Calculate isolation metric
            isolation_metric = np.sum(dists, axis=1)
            isolated_idx = np.argmax(isolation_metric)
            
            # Apply final expansion to the most isolated circle
            expansion_factor = (total_sum + 0.005 - total_sum) / (n - 1)
            new_radii = radii.copy()
            new_radii[isolated_idx] += expansion_factor * 1.2 + np.random.normal(0, 0.001)
            
            # Re-validate configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion factor for constraint enforcement
                new_radii = radii + (new_radii - radii) * 0.95

        v = res.x
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-11, "maxls": 100})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())