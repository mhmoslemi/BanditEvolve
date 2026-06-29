import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Generate initial randomized positions with geometric hashing for diverse cluster distributions
    xs = []
    ys = []
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        x_center = (col_idx + 0.5) / cols
        y_center = (row_idx + 0.5) / rows
        
        # Add geometric hash to disrupt regular grid patterns
        geometric_hash = np.random.rand(2) * 0.04
        x = x_center + geometric_hash[0] - 0.02
        y = y_center + geometric_hash[1] - 0.02
        
        # Staggered rows to enhance spatial diversity
        if row_idx % 2 == 1:
            x += 0.5 / cols * 0.95
        
        xs.append(np.clip(x, 0.0, 1.0))
        ys.append(np.clip(y, 0.0, 1.0))
    
    # Initialize radii with optimized base value considering grid spacing
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

    # Define constraints bounds for all circles
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    # Objective function to maximize sum of radii (minimize negative sum)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Generate constraints for boundaries and circle separations
    cons = []
    
    # Boundary constraints for all circles
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Circle separation constraints with vectorized geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            # Generate unique randomized geometric hashing for each pair
            hash_val = np.random.rand(2) * 0.03
            def constraint_func(v, i=i, j=j, hash_val=hash_val):
                dx = v[3*i] - v[3*j] + hash_val[0]
                dy = v[3*i+1] - v[3*j+1] + hash_val[1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # First optimization run with increased tolerance and iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Apply advanced adaptive perturbation to break local minima with geometric hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create perturbation matrix with geometric hashing for circle reconfiguration
        perturbation_vector = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        
        for i in range(n):
            perturbed_v[3*i] += perturbation_vector[i, 0]
            perturbed_v[3*i+1] += perturbation_vector[i, 1]
        
        # Second optimization run with enhanced configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion on the least constrained circle after spatial analysis
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
        
        min_dists = np.min(dist_matrix, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor based on spatial availability
        total_sum = np.sum(radii)
        expansion_factor = 0.008 / (n - 1)
        
        # Adjust radii with controlled expansion for least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with expanded radii and reconfiguration constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())