import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols  # 5 rows for 6 cols with 26 circles (24 in grid + 2 extra)
    # 3x3 grid with 9 cells (but we have 26 circles; we'll cluster into 4 groups of 6+6+6+8 or 8 layers of varying row counts)

    # Precompute spatial hashing with dynamic geometric scaling
    spatial_hash_seed = np.random.RandomState(np.random.randint(1, 1000000)).uniform(-1, 1, size=(n, 2))
    spatial_hash = spatial_hash_seed * 0.065  # Slightly larger range for more dynamic clustering
    
    # First-stage geometry: staggered grid with adaptive cluster centers
    xs = []
    ys = []
    
    # Group based on proximity to grid edges and create dynamic spatial hierarchy
    cluster_centers = []
    for i in range(n):
        layer = i // 4  # 26 divided into 6 layers, with 4 in early layers
        if layer < 2:
            col = i % 4
            row = (i // 4) + 0
            x = (col + 0.3 + spatial_hash[i,0]) / cols
            y = (row + 0.35 + spatial_hash[i,1])/ rows
        elif layer < 4:
            col = ((i - 4*(layer)) // 2) % cols # alternate column assignment
            row = (layer) + 0
            x = (col + 0.5 + spatial_hash[i,0]) / cols
            y = (row + 0.55 + spatial_hash[i,1]) / rows
        elif layer < 5:
            # 3D-like clustering
            col = (i - 4*(layer)) % cols
            row = (layer) + 0.1
            x = (col + 0.5 + spatial_hash[i,0] * np.sin(i)) / cols + 0.05
            y = (row + 0.5 + spatial_hash[i,1] * np.cos(i))
        else:
            # Final layer: staggered and overlapping
            col = (i - 4*5) % cols
            base_y = 0.75
            row = base_y + 0.15 + np.sqrt(i)
            x = (col + 0.45 + spatial_hash[i,0] * (1 - i/26)) / cols
            y = (row + spatial_hash[i,1] * (1 - i/26)) / rows
        # Apply spatial hashing distortion
        x += (spatial_hash[i,0] * np.exp(-i / 10))
        y += (spatial_hash[i,1] * np.exp(-i / 10))
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with adaptive scaling
    # Scale radii inversely with spatial density
    # First cluster: 6 circles with small radii
    # Second cluster: 7 circles with moderate radii
    # Third cluster: 8 circles with moderate radii
    # Fourth cluster: 5 circles with larger radii
    # Final cluster: 6 circles with largest radii
    # Use cluster-dependent initial values
    cluster_ids = []
    for i in range(n):
        # Determine cluster based on position in layered grid
        if i < 6: 
            cluster_ids.append(0)
        elif i < 13:
            cluster_ids.append(1)
        elif i < 21:
            cluster_ids.append(2)
        elif i < 25:
            cluster_ids.append(3)
        else:
            cluster_ids.append(4)
    
    # Adaptive cluster-based radii estimate: denser clusters get smaller radii
    cluster_radii = np.array([0.32, 0.35, 0.36, 0.375, 0.42]) # Larger clusters get smaller radius
    cluster_counts = np.array([6,7,8,5,6], dtype=int)
    
    # Compute per-circle initial radius based on cluster
    # Use a logarithmic scaling of cluster-based density to avoid clustering
    cluster_density = np.array([cluster_counts[i] for i in cluster_ids])
    cluster_radius_values = cluster_radii - (np.log(cluster_density + 1)) * 0.01
    cluster_radius_values = np.clip(cluster_radius_values, 1e-3, 0.5)
    r0 = cluster_radius_values[cluster_ids]
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0.copy()  

    # Ensure bounds list length matches
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries as required

    # Optimizer function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective: maximize sum of radii

    # Constraint construction with dynamic bounds (avoid lambda capture issues)
    cons = []

    # Add spatial bounds constraints
    for i in range(n):
        # 1. Left side: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # 2. Right side: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # 3. Bottom side: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # 4. Top side: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized constraint for overlaps (distance between any two circles) 
    # Use a more aggressive constraint with soft penalty for early iteration
    # Compute pairwise distances with vectorization to reduce recomputation
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with closure for i and j, but use partials to avoid capturing i,j in loops
            def constraint_func_fixed(i=i, j=j):
                def inner(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                return inner
            cons.append({"type": "ineq", "fun": constraint_func_fixed})

    # Initial optimization pass
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-8})

    # Iterative refinement with multiple passes:
    # 1. First pass: optimize using current positions and fixed cluster layout
    # 2. Second pass: geometric dissection and reconfiguration for dynamically interacting pairs
    # 3. Third pass: targeted expansion and topology reordering to break stalemate configurations

    if res.success:
        # Initial refined v after first pass
        v_current = res.x
        # Track progress of radius sum to detect plateau
        last_sum = np.sum(v_current[2::3])
        plateau_count = 0

        # First major pass with spatial hashing and adaptive perturbations
        for phase in range(3):
            # Apply multi-phase perturbation to spatial constraints
            # Phase 1: spatial hashing with increasing spatial variance
            # Phase 2: reconfiguration with random spatial distortion
            # Phase 3: geometric dissection for critical circle relationships
            if phase == 0:
                # Spatial hashing with geometric distortion
                spatial_perturb = np.random.rand(n, 2) * 0.04
                perturbed_v = v_current.copy()
                for i in range(n):
                    perturbed_v[3*i] += spatial_perturb[i,0] * ( (v_current[3*i+2] * 10) / np.mean(v_current[2::3]))
                    perturbed_v[3*i+1] += spatial_perturb[i,1] * ( (v_current[3*i+2] * 10) / np.mean(v_current[2::3]))
                # Second pass with perturbed vector
                res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-9})
            elif phase == 1:
                # Apply geometric dissection for dynamically interacting circles
                # Identify and isolate most dynamically interacting pairs
                # Use distance squared (more numerically stable than sqrt)
                centers = np.column_stack([v_current[0::3], v_current[1::3]])
                radii = v_current[2::3]

                # Compute distance matrix
                dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
                dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
                dists = (dx ** 2 + dy ** 2) ** .5
                interaction = (dists ** 2 - (radii[:, np.newaxis] + radii[np.newaxis, :]) ** 2) 
                # Use distance-squared minus overlap to find most interacting pairs
                interaction_matrix = np.sum(interaction, axis=1)
                top_pairs = np.argsort(interaction_matrix)[-5:]  # Get top 5 most dynamically interacting circles
                
                # Create new vector with adjusted positions of top 5 circles
                new_v = v_current.copy()
                for idx in top_pairs:
                    # Apply controlled random spatial shift
                    new_v[3*idx] += np.random.uniform(-0.03, 0.03)
                    new_v[3*idx+1] += np.random.uniform(-0.03, 0.03)
                    # Adjust radii with small stochastic expansion for critical circles
                    new_v[3*idx+2] += np.random.uniform(-0.005, 0.002)

                # Run constraint-based optimization on perturbed configuration
                res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
            else:
                # Final phase: adaptive radius expansion on least constrained circle
                centers_final = np.column_stack([v_current[0::3], v_current[1::3]])
                radii_final = v_current[2::3]
                # Compute distance matrix for final check
                dx = centers_final[:, np.newaxis, 0] - centers_final[np.newaxis, :, 0]
                dy = centers_final[:, np.newaxis, 1] - centers_final[np.newaxis, 1]
                dists = (dx ** 2 + dy ** 2)
                # Find least constrained circle by maximizing minimal inter-circle distance
                # Use distance squared to avoid sqrt for numerical stability
                min_dists = np.min(dists, axis=1)
                least_constrained_idx = np.argmax(min_dists)

                # Calculate expansion based on current radius sum and spatial feasibility
                # Use adaptive expansion that is 1.2x base rate with a small randomness factor
                current_total = np.sum(radii_final)
                expansion_factor = 0.003 * (1 + np.random.rand() * 0.2)
                new_radii = radii_final.copy()
                new_radii[least_constrained_idx] += expansion_factor * 1.2

                # Apply expansion while maintaining feasibility
                while True:
                    # Try new radii
                    new_v = v_current.copy()
                    new_v[2::3] = new_radii
                    # Ensure no negative radii
                    new_v[2::3] = np.clip(new_v[2::3], 1e-5, 0.5)
                    # Check feasibility
                    centers_expanded = np.column_stack([new_v[0::3], new_v[1::3]])
                    valid = True
                    for i in range(n):
                        for j in range(i+1, n):
                            dist = np.sqrt((centers_expanded[i, 0] - centers_expanded[j, 0])**2 + 
                                           (centers_expanded[i, 1] - centers_expanded[j, 1])**2)
                            if dist <= new_v[3*i+2] + new_v[3*j+2] - 1e-12:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        break
                    else:
                        # Back off slightly and redistribute
                        new_radii = radii_final + (new_radii - radii_final) * 0.95

                # Run final optimization pass
                res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
            
            v_current = res.x if res.success else v_current
            current_sum = np.sum(v_current[2::3])
            if np.abs(current_sum - last_sum) < 1e-4 and plateau_count > 5:
                # If plateauing, terminate
                break
            last_sum = current_sum
            plateau_count = 0
        # Final check for validity of result
        v = res.x if res.success else v0
    else:
        v = v0

    # Final validation and output
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-5, 0.5)
    # Final double-check to avoid overexpansion in case of numerical issues
    # Recompute distances with final configuration
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            # If out of bounds, force boundary constraint
            v[3*i] = np.clip(x, 0.0, 1.0)
            v[3*i+1] = np.clip(y, 0.0, 1.0)
            if (abs(x) > 1.0 + 1e-12 or abs(y) > 1.0 + 1e-12):
                v[3*i+2] = max(1e-5, (1.0 - (v[3*i] if v[3*i] < 1.0 else 0.0)) / 2)
            radii[i] = v[3*i+2]
    
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= radii[i] + radii[j] - 1e-12:
                # If overlap, reduce radius of one
                # Favor reducing the smaller one
                if radii[i] < radii[j]:
                    radii[i] -= 0.0001
                else:
                    radii[j] -= 0.0001
                # Ensure not below minimum
                radii = np.clip(radii, 1e-5, 0.5)
                # Update centers for radii change
                centers = np.column_stack([v[0::3], v[1::3]])
    
    # After cleanup, clip and ensure final radii
    radii = np.clip(radii, 1e-5, 0.5)
    return centers, radii, float(radii.sum())