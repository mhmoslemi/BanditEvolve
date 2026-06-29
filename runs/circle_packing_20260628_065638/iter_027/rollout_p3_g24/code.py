import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Spatial hashing initialization and asymmetric initialization
    xs = []
    ys = []
    # Initialize using a hybrid geometric hashing and staggered grid
    for i in range(n):
        row = i // cols
        col = i % cols
        col_center = (col + 0.5) / cols
        row_center = (row + 0.5) / rows

        # Add noise to break symmetry and avoid clustering
        noise_x = np.random.uniform(-0.04, 0.04)
        noise_y = np.random.uniform(-0.04, 0.04)
        x = col_center + noise_x
        y = row_center + noise_y
        
        # Apply staggered vertical offset
        if row % 2 == 1:
            x += 0.5 / cols * (1 + np.random.rand() * 0.2)  # Add stochastic stagger
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with spatial hashing
    r0 = 0.34 / cols * (1 + (np.random.rand(n) * 0.15 - 0.075))  # Slight variance in initial radius
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Bounds for the entire decision vector
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)])  # Always 3*n entries

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Generate constraints - use lambda with closures
    cons = []

    # Boundary constraints
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints with lambda closures
    for i in range(n):
        for j in range(i+1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with enhanced convergence strategies
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})

    # Apply non-local geometric reconfiguration using advanced spatial hashing and constraint mutation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Create geometric hash with adaptive spatial scaling for non-local configuration
        spatial_hash = np.random.rand(n, 2) * 0.05  # Reduced perturbation
        perturbed_v = v.copy()
        for i in range(n):
            # Scale spatial hashing based on current radius for better configuration
            scale_factor = 1 + (radii[i] / (n * 0.05))  # Dynamic scaling
            perturbed_v[3*i] += spatial_hash[i, 0] * scale_factor
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale_factor
        
        # Re-evaluate with non-local spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Perform targeted radius expansion on the least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute pairwise proximity matrix and identify least constrained circle
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Initialize expansion with spatial hashing to target underutilized space
        directional_hash = np.random.rand(n, 2) * 0.05 - 0.025  # Random direction for expansion
        expansion_factor_base = 0.008 / (n - 1) * (np.std(radii) / np.std(radii))  # Stable base growth

        # Apply directional expansion with stochastic component
        new_radii = radii.copy()
        # Add over-expansion to most constrained circles
        new_radii[least_constrained_idx] += expansion_factor_base * 1.5
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor_base * (1.0 + directional_hash[i, 0] * 0.4)
        
        # Apply expansion and validate
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])

            # Validate configuration
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
                # Reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Final configuration with expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())