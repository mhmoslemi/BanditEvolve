import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Improved initialization: use more structured random sampling
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Define a grid with intentional spacing
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        
        # Add small random perturbations to spread circles out
        x_offset = np.random.uniform(-0.07, 0.07)
        y_offset = np.random.uniform(-0.07, 0.07)
        x_perturb = x_offset * (1.0 + (row % 2) * 0.2)  # stagger rows further
        y_perturb = y_offset * (1.0 - (row % 2) * 0.2)  # stagger rows less
        
        x_center = x_base + x_perturb
        y_center = y_base + y_perturb
        
        xs.append(x_center)
        ys.append(y_center)
    
    r0 = 0.4 / cols - 1e-3  # Slightly increase initial radius
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n, matches v

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            # Use a vectorized lambda to avoid function redefinition
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerances and moderate iteration count
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-10})

    # Asymmetric reconfiguration: stochastic spatial perturbation
    if res.success:
        v = res.x
        # Apply randomized geometric hashing to disrupt current configuration
        random_hash = np.random.rand(n, 2) * 0.05  # Reduced perturbation strength
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion with dynamic selection
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]  # x differences
                dy = v[3*i+1] - v[3*j+1]  # y differences
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Compute isolation metric (smallest sum of reciprocals of distances)
        isolation = np.sum(1 / (dists + 1e-8), axis=1)
        isolated_idx = np.argmin(isolation)
        
        # Calculate controlled expansion amount
        total_sum = np.sum(radii)
        expansion_factor = 0.0065 / (n - 1)  # Slightly increased target
        
        # Adjust radii for isolated circle and neighbors
        new_radii = radii.copy()
        new_radii[isolated_idx] += expansion_factor * 1.2  # Overexpansion to trigger configuration shift
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion_factor * 0.9  # Reduce expansion for neighboring circles
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final fallback to initial configuration
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())