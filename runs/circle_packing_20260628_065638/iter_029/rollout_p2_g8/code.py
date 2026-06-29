import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    cols2 = 5  # Secondary grid for fine-grained spatial exploration
    
    # Seed optimization for consistent, repeatable high-quality results
    np.random.seed(2024) # Fixed for repeatability & reproducibility

    def create_initial_centers(grid_cols, grid_rows, perturbation_range=0.04):
        xs, ys = [], []
        for i in range(n):
            col = i % grid_cols
            row = i // grid_cols
            # Primary grid
            x_cent = (col + 0.5) / grid_cols
            y_cent = (row + 0.5) / grid_rows
            # Randomized offset
            x = x_cent + np.random.uniform(-perturbation_range, perturbation_range)
            y = y_cent + np.random.uniform(-perturbation_range, perturbation_range)
            # Staggered rows to avoid alignment
            if row % 2 == 1:
                x += 0.5 / grid_cols
            if np.random.rand() < 0.3: # With 30% chance, introduce non-standard perturbation
                x += np.random.uniform(-0.1, 0.1) 
                y += np.random.uniform(-0.1, 0.1)
            xs.append(np.clip(x, 0, 1))
            ys.append(np.clip(y, 0, 1))
        return xs, ys

    # Multi-layered grid setup with dual optimization paths
    # Layer 1: Primary grid with dynamic perturbation
    xs1, ys1 = create_initial_centers(cols, rows, 0.04)
    
    # Layer 2: Secondary grid with stricter spacing for tighter packing
    xs2, ys2 = create_initial_centers(cols2, rows, 0.025) 
    xs2 = [(x + np.random.uniform(-0.04, 0.04)) for x in xs2]
    ys2 = [(y + np.random.uniform(-0.04, 0.04)) for y in ys2]
    
    # Hybrid strategy: combine primary and secondary grid configurations
    xs = xs1[:]
    ys = ys1[:]
    if np.random.rand() < 0.5:
        # Swap with secondary configuration with random permutation
        permutation = np.random.permutation(n)
        xs = np.array(xs2)[permutation].tolist()
        ys = np.array(ys2)[permutation].tolist()
    
    # Create base radius: dynamic spacing-aware value
    # Base radius based on grid spacing and perturbation
    radius_base = (0.32 / cols) * 1.2  # Base radius with 20% improvement over baseline
    r0 = radius_base - 0.0002  # Slight adjustment to prevent edge clashes

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Ensure bounds list has exactly 3*n entries
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint generation with vectorized and efficient lambda closures
    # Note: Lambda captures are now done correctly with parameterized i and j
    cons = []

    # Boundary constraints using functional capture with lambda i
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
    
    # Circle-circle constraints with parameterized lambda for i and j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j])**2 + 
                    (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2
                )
            })

    # Initial optimization with high precision and iterations
    res = minimize(
        neg_sum_radii, v0, method='SLSQP',
        bounds=bounds, constraints=cons,
        options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-11, 'disp': False}
    )

    # Fallback in case of failure in primary optimization path
    if not res.success:
        print("Initial optimization failed. Reinitializing and reoptimizing.")
        
        # Hybrid grid with spatial hashing and perturbation
        xs = []
        ys = []
        for i in range(n):
            col = i % cols
            row = i // cols
            x_cent = (col + 0.5) / cols
            y_cent = (row + 0.5) / rows
            # Perturb with spatial awareness
            x = x_cent + np.random.uniform(-0.03, 0.03)
            y = y_cent + np.random.uniform(-0.03, 0.03)
            if row % 2 == 1:
                x += 0.5 / cols
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            xs.append(x)
            ys.append(y)
        
        v0 = np.empty(3 * n)
        v0[0::3] = np.array(xs)
        v0[1::3] = np.array(ys)
        v0[2::3] = np.full(n, 0.33 / cols) # More aggressive base radius

        # Perform secondary optimization with updated v0
        res = minimize(
            neg_sum_radii, v0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': 3500, 'ftol': 1e-12, 'gtol': 1e-11}
        )

    # Apply gradient-aware spatial perturbation only on successful optimization
    if res.success:
        print("Optimization successful. Applying gradient-aware spatial reconfiguration.")
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Spatial-aware perturbation: directional amplification
        # Calculate spatial hashing with adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.12
        perturbed_v = v.copy()
        for i in range(n):
            dx_perturb = spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * 1.1
            dy_perturb = spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * 1.1
            perturbed_v[3*i] += dx_perturb
            perturbed_v[3*i+1] += dy_perturb
        
        # Re-optimization with perturbed state
        res = minimize(
            neg_sum_radii, perturbed_v, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-11}
        )

    # Post-optimization targeted expansion of least constrained circle
    if res.success:
        print("Targeted expansion initiated.")
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized distance matrix for constraint validation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate growth based on current total sum and potential for expansion
        current_total = np.sum(radii)
        # Targeting a growth of 0.0065, slightly above the previous state
        target_growth = 0.0065
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))

        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Add randomness to avoid deterministic behavior
                expansion_i = expansion_factor * (1.0 + 0.1 * np.random.rand())
                new_radii[i] += expansion_i

        # Apply expansion with constraint validation using vectorized approach
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])

            # Efficient validation using vectorized distance matrix
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
                # If invalid, decrease expansion by 5%
                new_radii = radii + (new_radii - radii) * 0.95

        # Update vector with expanded radii and re-optimize
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(
            neg_sum_radii, v_new, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-11}
        )

    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Ensure no numerical instabilities
    radii[radii < 1e-6] = 1e-6
    return centers, radii, float(radii.sum())