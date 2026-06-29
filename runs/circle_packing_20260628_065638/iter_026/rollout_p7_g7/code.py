import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Use a more sophisticated initialization with adaptive grid and spatial randomness
    xs = []
    ys = []
    base_grid = np.zeros((rows, cols))
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce adaptive perturbation based on grid structure
        x_perturb = np.random.uniform(-0.05, 0.05) * (1 + 0.2 * np.sin(row * np.pi / rows))
        y_perturb = np.random.uniform(-0.05, 0.05) * (1 + 0.2 * np.cos(col * np.pi / cols))
        x = x_center + x_perturb
        y = y_center + y_perturb
        # Create staggered grid with row-based horizontal offset
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Use a more refined initial radius based on grid density
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with proper closure handling
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with more efficient closure handling
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with default arguments to avoid capture issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with adaptive max iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})

    # Asymmetric reconfiguration: random spatial perturbation with adaptive expansion
    if res.success:
        v = res.x
        # Compute current radii and centers
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute circle constraint map
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_matrix[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Compute constraint slack for each circle
        constraint_slack = np.zeros(n)
        for i in range(n):
            constraint_slack[i] = np.min(dist_matrix[i, i+1:])
        
        # Find least constrained circle (smallest constraint slack)
        least_constrained_idx = np.argmin(constraint_slack)
        
        # Add asymmetric spatial perturbation with adaptive amplitude
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (1 + 0.2 * np.random.rand())
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (1 + 0.2 * np.random.rand())
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Apply targeted radius expansion on least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_matrix[i, j] = np.sqrt(dx*dx + dy*dy)
        
        constraint_slack = np.zeros(n)
        for i in range(n):
            constraint_slack[i] = np.min(dist_matrix[i, i+1:])
        least_constrained_idx = np.argmin(constraint_slack)
        
        # Compute target radius sum through controlled expansion
        target_total_sum = np.sum(radii) + 0.006
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Create expansion vector
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Over-expand slightly
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())

        # Validated expansion with soft constraints
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            is_valid = True
            
            # Check for overlaps and boundary constraints
            for i in range(n):
                # Boundary checks
                if (expanded_centers[i, 0] - new_radii[i] < 0 or
                    expanded_centers[i, 0] + new_radii[i] > 1 or
                    expanded_centers[i, 1] - new_radii[i] < 0 or
                    expanded_centers[i, 1] + new_radii[i] > 1):
                    is_valid = False
                    break
                
                # Overlap checks
                for j in range(i+1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if is_valid:
                break
            else:
                # Reduce expansion by 10% if invalid
                new_radii = radii + (new_radii - radii) * 0.95

        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded configuration and optimized constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())