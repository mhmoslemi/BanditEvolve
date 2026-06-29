import numpy as np

def run_packing():
    """
    Implements an improved circle packing method using a combination 
    of geometric hashing, dynamic gradient approximation, and targeted expansion.
    """
    n = 26
    cols = int(np.ceil(np.sqrt(n)))  # More adaptive col/row grid
    rows = (n + cols - 1) // cols  # Ensure proper row allocation

    # Initialize positions with asymmetric spacing and advanced geometric hash
    xs = [0.0] * n
    ys = [0.0] * n
    for i in range(n):
        row = i // cols
        col = i % cols
        # Adaptive center position based on row and column
        max_x = 1.0 - 2.0 / rows * row
        max_y = 1.0 - 2.0 / cols * col
        x_center = (col + 0.5) / cols + (np.random.rand() - 0.5) * (1.0 - 2.0 / cols)
        y_center = (row + 0.5) / rows + (np.random.rand() - 0.5) * (1.0 - 2.0 / rows)
        # Spatial perturbation using geometric hashing
        hash_map = np.random.rand(n, 2) * 0.1
        x = x_center + hash_map[i, 0] * (1.0 - 2.0 / cols) * np.cos(np.random.rand() * 2 * np.pi)
        y = y_center + hash_map[i, 1] * (1.0 - 2.0 / rows) * np.sin(np.random.rand() * 2 * np.pi)
        ys[i] = y
        xs[i] = x

    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds with consistent length and tighter radius limits
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint functions with proper closure handling
    cons = []

    # Add boundary constraints with vectorized evaluation
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Add pairwise distance constraints with vectorization
    for i in range(n):
        for j in range(i+1, n):
            # Use lambda with captured i,j to avoid closure capture
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization stage with dynamic gradient approximation
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "eps": 1e-12, "gtol": 1e-9})

    # Asymmetric spatial reconfiguration with dynamic hashing
    if res.success:
        v = res.x
        # Create spatial hash for reconfiguration with adaptive parameters
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Apply asymmetric scaling based on radius for reconfiguration 
            scaling = (v[3*i+2] / np.mean(v[2::3])) * 0.75
            perturbed_v[3*i] += spatial_hash[i, 0] * scaling
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scaling
        # Second optimization stage with tighter tolerance
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-13, "gtol": 1e-10})

    # Targeted radius expansion on least constrained circle with spatial analysis
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute spatial influence matrix and distance metrics
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Identify least constrained circle by maximizing spatial distance margin
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate expansion based on current state
        current_total = np.sum(radii)
        target_growth = 0.006
        expansion_factor = (target_growth / (n - 1)) * (current_total / np.sum(radii))
        
        # Apply asymmetric expansion with spatial-aware redistribution
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Over-expansion for exploration
        for i in range(n):
            if i != least_constrained_idx:
                # Stochastic expansion with dynamic bias for constrained circles
                if i in range(0, n//2):  # Prioritize non-edge circles
                    expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())
                else:
                    expansion_i = expansion_factor * (0.8 + 0.1 * np.random.rand())
                new_radii[i] += expansion_i

        # Apply expansion safely with validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx2 = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy2 = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx2**2 + dy2**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # If invalid, reduce expansion slightly in a controlled way
                new_radii = radii + (new_radii - radii) * 0.95

        v_new = v.copy()
        v_new[2::3] = new_radii
        # Final optimization with tight tolerances
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-13, "gtol": 1e-10})

    # Final configuration check
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())