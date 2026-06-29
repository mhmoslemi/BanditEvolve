import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols

    # Initialize positions with staggered hexagonal grid, randomized offsets 
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Staggered pattern for even rows
        if row % 2 == 1:
            base_x += 0.5 / cols  # shifted in x for staggered row

        # Add controlled randomness to avoid symmetry
        x_rand = np.random.uniform(-0.04, 0.04)
        y_rand = np.random.uniform(-0.02, 0.02)
        
        x = base_x + x_rand
        y = base_y + y_rand
        xs.append(x)
        ys.append(y)
    
    # Adaptive base radius calculation with grid geometry
    # Hex grid efficient packing distance: sqrt(3) / (2 * cols) * 0.95
    r0 = 0.75 / cols * 0.75 - 1e-3  # Conservative with buffer
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Ensure bounds list has 3*n entries
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint functions with i-aware captures
    cons = []
    for i in range(n):
        # Left: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right: 1 - (x_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top: 1 - (y_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized pairwise overlap constraints using precomputed indices
    # Use np.triu to only generate upper triangle for efficiency
    for i in range(n):
        for j in range(i+1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                         - (v[3*i+2] + v[3*j+2])**2})

    # First optimization phase: base grid with initial spacing (SLSQP with tight tolerances)
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11, 
                                             "eps": 1e-10, "disp": False})

    if res.success:
        # Extract optimal state
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Spatial hashing for asymmetric reconfiguration
        # Generate directional hashing weighted by radius
        # This introduces a new dimension of spatial manipulation
        spatial_hash = np.random.rand(n, 2) * 0.05
        directional_hash = np.random.rand(n, 2) * 0.08

        # Apply spatial perturbation with directionality and radius scaling
        # This allows controlled asymmetry for better expansion potential
        perturbed_v = v.copy()
        for i in range(n):
            # Spatial perturbation
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i]/np.mean(radii)) * 1.1
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i]/np.mean(radii)) * 1.1
            # Directional perturbation
            perturbed_v[3*i+2] += directional_hash[i, 0] * 0.002 * (1 + 0.75 * np.sqrt(radii[i]))
        
        # Second optimization: reconfigure with asymmetric spatial hashing
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 350, "ftol": 2e-11, 
                                                 "eps": 1e-10, "disp": False})

        if res.success:
            # Extract refined state
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])

            # Compute distance matrix
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)

            # Identify two most dynamically interacting circles
            # Find the two pairs with minimal spacing (dynamic interaction)
            interaction_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    interaction_matrix[i, j] = dists[i,j] - (radii[i] + radii[j]) + 1e-12

            # Pairwise interaction energy for each pair
            # The pair with the lowest margin of safety is the most dynamically interacting
            # Sort to find top two interacting pairs
            sorted_indices = np.argpartition(interaction_matrix, -2, axis=None)[-2:]

            # Convert flat index to 2D
            dynamic_pair1 = sorted_indices // n
            dynamic_pair2 = sorted_indices % n

            dynamic_pair1 = (dynamic_pair1[0], dynamic_pair2[0])
            dynamic_pair2 = (dynamic_pair1[1], dynamic_pair2[1])

            # Extract these two circles as primary reconfiguration targets
            i1, j1 = dynamic_pair1
            i2, j2 = dynamic_pair2

            # Generate new spatial vector with directional hashing and radius scaling
            # Apply directional hashing and radius-aware spatial displacement
            # Also introduce soft constraints for these targeted pairs
            # Use soft constraints to allow subtle spatial readjustments

            spatial_hash = np.random.rand(n, 2) * 0.05
            directional_hash = np.random.rand(n, 2) * 0.05

            v_new = v.copy()

            for i in range(n):
                x, y, r = v[i*3], v[i*3+1], v[i*3+2]
                spatial_x = spatial_hash[i, 0] * (r / np.mean(radii)) * 0.75
                spatial_y = spatial_hash[i, 1] * (r / np.mean(radii)) * 0.75
                direction_x = directional_hash[i, 0] * 0.002 * (1 + 0.5 * np.sqrt(r))
                direction_y = directional_hash[i, 1] * 0.002 * (1 + 0.5 * np.sqrt(r))

                # Apply perturbation but not to the two most dynamic circles
                if i not in [i1, j1, i2, j2]:
                    v_new[i*3] += spatial_x + direction_x
                    v_new[i*3+1] += spatial_y + direction_y
                    v_new[i*3+2] += 0.001 * (np.random.rand() - 0.5) * (1 + 0.3 * np.sqrt(r))

            # Re-configure these two circles to promote new spatial relationships
            # Add directional expansion bias to these two to encourage more spacing
            expansion_factor = 0.005 * (np.sum(radii) / np.mean(radii)) * 1.1

            # Add soft constraints to encourage expansion in targeted circles
            soft_constraints = []
            for i in [i1, j1, i2, j2]:
                soft_constraints.append({"type": "ineq",
                                         "fun": lambda v, i=i: v[3*i+2] + 0.005 * np.random.rand() * 1.3})
            
            # Third optimization phase: spatial reconfiguration and expansion on targeted
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons + soft_constraints, options={"maxiter": 400, "ftol": 1e-11, 
                                                                    "eps": 1e-10, "disp": False})
        
        if res.success:
            # Final configuration
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])

            # Final validation and error handling
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                valid, reason = validate_packing(centers, radii)
            
            if not valid:
                # Fallback after reconfiguration
                return run_packing()
    
    # Final fallback
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())