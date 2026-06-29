import numpy as np

def run_packing():
    n = 26
    # Geometric tiling with randomized clustering to avoid symmetry
    cols = 5
    rows = (n + cols - 1) // cols
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        # Base grid positions
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        # Randomize offset and apply asymmetric staggering
        x_offset = np.random.uniform(-0.03, 0.03)
        y_offset = np.random.uniform(-0.03, 0.03)
        x = x_base + x_offset
        y = y_base + y_offset
        # Add stagger for alternating rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radii estimation, adjusted for grid spacing
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Define bounds for positions and radii
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraints for boundary conditions (x, y, r)
    cons = []
    for i in range(n):
        # left: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # right: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # bottom: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # top: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Fast overlap constraint calculation using vectorization
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization step with high tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})

    # Post-optimization refinement with perturbation of least constrained circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute minimum distances to all circles
        dists = np.zeros((n, n))
        for i in range(n):
            dx = centers[:, 0] - centers[i, 0]
            dy = centers[:, 1] - centers[i, 1]
            dists[i] = np.sqrt(dx**2 + dy**2)
        
        # Find indices of least constrained circles
        min_dists = np.min(dists, axis=1)
        least_constrained_indices = np.argsort(min_dists)[-2:]  # Select top 2 least constrained
        
        # Add small randomized perturbations to those circles
        for i in least_constrained_indices:
            v[3*i] += np.random.uniform(-0.03, 0.03)
            v[3*i+1] += np.random.uniform(-0.03, 0.03)
            v[3*i+2] += np.random.uniform(-0.003, 0.003)
        
        # Re-optimization with refined positions
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    # Final refinement by expanding the most under-constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Recompute minimum distances and select the most constrained circle
        dists = np.zeros((n, n))
        for i in range(n):
            dx = centers[:, 0] - centers[i, 0]
            dy = centers[:, 1] - centers[i, 1]
            dists[i] = np.sqrt(dx**2 + dy**2)
        
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor relative to sum and constraint
        max_additional_sum = 0.008
        expansion_factor = (max_additional_sum - (radii[least_constrained_idx] - 1e-4)) / (n - 1)

        # Adjust radii with targeted expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.4
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Apply new radii and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final check and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())