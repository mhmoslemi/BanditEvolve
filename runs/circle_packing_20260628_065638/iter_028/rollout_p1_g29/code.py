import numpy as np

def run_packing():
    n = 26
    # Create a grid with dynamic sizing, allowing more optimal layout
    cols = 5
    rows = (n + cols - 1) // cols
    num_cols_for_initial = cols
    num_rows_for_initial = rows

    # Initialize positions using a randomized hexagonal grid pattern with dynamic offset
    xs = []
    ys = []
    for i in range(n):
        col = i % num_cols_for_initial
        row = i // num_cols_for_initial
        base_x = (col + 0.5) / num_cols_for_initial
        base_y = (row + 0.5) / num_rows_for_initial

        # Staggered offset for hexagonal packing
        if row % 2 == 1:
            x_offset = 0.5 / num_cols_for_initial
        else:
            x_offset = 0.0

        # Apply randomized perturbation to break symmetry
        x_rand = np.random.uniform(-0.06, 0.06)
        y_rand = np.random.uniform(-0.06, 0.06)
        x = base_x + x_offset + x_rand
        y = base_y + y_rand

        # Ensure boundaries (with small margin)
        x = np.clip(x, 1e-12, 1.0 - 1e-12)
        y = np.clip(y, 1e-12, 1.0 - 1e-12)
        xs.append(x)
        ys.append(y)

    r0 = (0.45 / num_rows_for_initial) - 1e-3  # Slightly larger base radius than previous
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3n matches decision vector

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with direct lambda closures and i capture
    cons = []
    for i in range(n):
        # Left margin (x - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right margin (x + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom margin (y - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top margin (y + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlapping constraint with direct lambda with i,j capture
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j])**2 + 
                    (v[3*i+1] - v[3*j+1])**2 -
                    (v[3*i+2] + v[3*j+2])**2
                )
            })

    # Initial optimization with adaptive tolerance and max iterations
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds, 
        constraints=cons, 
        options={"maxiter": 1200, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-9, "disp": False}
    )

    # Dynamic refinement: apply spatial perturbation and re-evaluation
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Spatial hashing for directional perturbation
        spatial_hash = np.random.rand(n, 2) * 0.03  # Reduced perturbation magnitude
        
        # Apply directional spatial perturbation based on spatial hashing and relative magnitudes
        perturbed_v = v.copy()
        for i in range(n):
            # Perturb spatial positions proportional to radius
            radius_ratio = radii[i] / np.mean(radii)
            dx = spatial_hash[i, 0] * radius_ratio * 0.01  # Small directional shift
            dy = spatial_hash[i, 1] * radius_ratio * 0.01
            perturbed_v[3*i] += dx
            perturbed_v[3*i+1] += dy

        # Re-evaluate with perturbed geometry
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP", 
            bounds=bounds, 
            constraints=cons, 
            options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-9, "disp": False}
        )

    # Dynamic expansion on least constrained circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Calculate pairwise distances with vectorized approach
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Identify least constrained circle (max of min distances)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate potential expansion to target the highest radii sum
        current_sum = np.sum(radii)
        target_sum = current_sum + 0.008
        max_possible_increment = target_sum - current_sum  # Max possible expansion
        expansion_factor = max_possible_increment / n * 1.2  # Allow some overflow for better configuration

        # Create directional expansion vector with focus on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.35  # More aggressive expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Expand with directional component based on relative position to least constraining circle
                dx_rel = centers[least_constrained_idx, 0] - centers[i, 0]
                dy_rel = centers[least_constrained_idx, 1] - centers[i, 1]
                dist_to_least = np.linalg.norm([dx_rel, dy_rel])
                if dist_to_least < 0.25:  # Near neighbor
                    expansion = expansion_factor * 2.0  # Boost for nearby circles
                else:
                    expansion = expansion_factor * 1.0
                new_radii[i] += expansion * (1.0 + np.random.rand() * 0.1)  # Stochastic addition for exploration

        # Apply expansion with constraint validation
        while True:
            # Create expanded vector
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])

            # Validate expanded configuration
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
                # If invalid, decrease expansion gradually
                new_radii = radii + (new_radii - radii) * 0.95  # Moderate reduction strategy

        # Update the optimization vector and proceed
        v = expanded_v

    # Final evaluation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())