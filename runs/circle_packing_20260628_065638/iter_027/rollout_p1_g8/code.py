import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Generate randomized staggered hexagonal lattice grid for 1st seed configuration
    def generate_staggered_seed():
        xs = []
        ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            base_x = (col + 0.5) / cols
            base_y = (row + 0.5) / rows
            x = base_x + np.random.uniform(-0.05, 0.05)
            y = base_y + np.random.uniform(-0.05, 0.05)
            # Offset alternate rows in staggered hexagonal grid
            if row % 2 == 1:
                x += 0.25 / cols
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    # Generate geometrically perturbed version for 2nd seed configuration
    def generate_perturbed_seed():
        xs_seed, ys_seed = generate_staggered_seed()
        # Apply geometric perturbation based on circle radii
        perturbation_factor = 0.08 / cols
        perturbation = np.random.rand(n, 2) * perturbation_factor
        x_perturb = perturbation[:, 0]
        y_perturb = perturbation[:, 1]
        return xs_seed + x_perturb, ys_seed + y_perturb

    # Use two distinct seed configurations for more robust optimization
    xs1, ys1 = generate_staggered_seed()
    xs2, ys2 = generate_perturbed_seed()

    # Choose initial seed configuration based on radius space
    # Choose the one that minimizes radius sum with spatial constraints
    r0 = 0.4 / cols - 1e-3
    v0_1 = np.concatenate([xs1, ys1, np.full(n, r0)])
    v0_2 = np.concatenate([xs2, ys2, np.full(n, r0)])
    
    # Choose seed with potential for higher initial radius sum
    seed_candidates = [v0_1, v0_2]
    seed_indices = range(2)
    
    # Calculate potential initial radius sum for both seeds
    potential_r1 = np.sum(np.min(np.sqrt(
        (xs1[:, np.newaxis] - xs1[np.newaxis, :])**2 + 
        (ys1[:, np.newaxis] - ys1[np.newaxis, :])**2), axis=1)) * r0
    potential_r2 = np.sum(np.min(np.sqrt(
        (xs2[:, np.newaxis] - xs2[np.newaxis, :])**2 + 
        (ys2[:, np.newaxis] - ys2[np.newaxis, :])**2), axis=1)) * r0
    initial_seed = seed_candidates[np.argmax([potential_r1, potential_r2])]

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n entries for 26 circles

    # Vectorized constraint definitions with lambda closures
    def define_constraints():
        cons = []
        for i in range(n):
            # Left boundary constraint: x_i - r_i >= 0
            cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
            # Right boundary constraint: 1 - x_i - r_i >= 0
            cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
            # Bottom boundary constraint: y_i - r_i >= 0
            cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
            # Top boundary constraint: 1 - y_i - r_i >= 0
            cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                # Function to evaluate distance^2 - (r_i + r_j)^2
                cons.append({"type": "ineq", 
                             "fun": (lambda v, i=i, j=j: 
                                     (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                     - (v[3*i+2] + v[3*j+2])**2)})
        return cons

    cons = define_constraints()

    # First optimization phase with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, initial_seed, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "eps": 1e-12})

    if not res.success:
        # Try different seed with better perturbation
        v0 = np.concatenate([xs2, ys2, np.full(n, r0)])
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "eps": 1e-12})

    # Second optimization run with fine-tuned configuration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply spatial hashing with adaptive scaling based on cluster density
        cluster_density = np.zeros(n)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        dists[dists < 1e-8] = 1e-8  # Avoid division by zero
        avg_dist_per_circle = np.mean(1 / dists)
        cluster_density = 1 / (np.min(1 / dists, axis=1) * avg_dist_per_circle)

        # Generate spatial perturbation based on cluster density
        spatial_hash = np.random.rand(n, 2)
        spatial_perturb = spatial_hash * (0.04 / cols) * (1 + cluster_density / np.max(cluster_density))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_perturb[i, 0]
            perturbed_v[3*i+1] += spatial_perturb[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-12})

    # Third step: identify least constrained circle and perform intelligent expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify the least constrained circle (maximin distance to others)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate target expansion based on current radius sum and topology-aware growth
        current_total = np.sum(radii)
        max_possible_expansion = 0.004  # Conservative estimate
        expansion_target = current_total + max_possible_expansion
        expansion_factor = expansion_target / (n)  # Base expansion per circle

        # Create expansion vector with targeted expansion on least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.25  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.15 * np.random.rand())  # Stochastic expansion
                new_radii[i] += expansion_i

        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
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
                # If invalid, scale down expansion by 2-5% incrementally
                new_radii = radii + (new_radii - radii) * 0.93
        
        # Final optimization pass with tight tolerances and adaptive constraints
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 450, "ftol": 1e-11, "eps": 1e-12})

    # Final fallback if any optimization failed
    v = res.x if res.success else initial_seed
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())