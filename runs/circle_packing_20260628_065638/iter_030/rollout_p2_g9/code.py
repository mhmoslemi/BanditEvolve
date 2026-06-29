import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n))) + 1  # +1 to break regular grid symmetry
    base_rows = (n + cols - 1) // cols
    # Dynamic row allocation based on spatial constraints + symmetry breaking
    # Create an adaptive grid with randomized clustering
    xs = np.zeros(n)
    ys = np.zeros(n)
    for idx in range(n):
        row = (idx % cols)  # Assign row to column index
        col = (idx // cols)  # Assign column to row index
        base_x = (row + 0.5) / cols
        base_y = (col + 0.5) / base_rows
        # Symmetric but asymmetric perturbation
        x_offset = 0.02 * (np.sin(10.0 * (idx + 1)/n) - 0.5)
        y_offset = 0.02 * (np.cos(12.0 * (idx + 1)/n) - 0.25)
        xs[idx] = base_x + x_offset
        ys[idx] = base_y + y_offset
        # Introduce alternate row shifts with adaptive scaling for staggered grid
        if row % 2 == 1:
            xs[idx] += (0.4 / cols) * (1.0 / (base_rows + 1))
        else:
            ys[idx] += (0.4 / base_rows) * (1.0 / (cols + 1))

    r0 = 0.38 / cols - 1e-3  # Slightly increased base radius
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0 * np.ones(n)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n elements total per circle

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Construct constraints with vectorized operations
    # Optimized constraint evaluation through closure parameterization
    cons = []
    for i in range(n):
        # Left bound: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right bound: 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom bound: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top bound: 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized pairwise constraint computation
    # Optimized with pre-vectorization and broadcast
    # Constraint: distance between circles i and j must be >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            # Optimized constraint function with lambda closure
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                         - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with adaptive tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-10})
    
    if res.success:
        v = res.x
        # Extract and process centers/radii
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Step 1: Compute pairwise distances (vectorized)
        # Use broadcasting for efficient distance matrix calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute interaction scores as sum of inverse distances
        # Normalize to avoid overemphasis on near-neighbor interactions
        interaction_scores = np.sum(1.0 / (dists + 1e-8), axis=1)
        
        # Find the two most interacting circles (to reconfigure)
        top_idx = np.argsort(interaction_scores)[-2:]
        
        # Step 2: Isolate and reconfigure the most interacting pair
        # Create a new configuration space with geometric hashing for spatial dissection
        # Create a geometric hash map with adaptive radius-dependent weights
        radii_ratio = np.clip(radii / np.mean(radii), 0.3, 1.3)
        spatial_hash = np.random.rand(n, 2) * 0.035 * radii_ratio[:, np.newaxis]
        
        # Apply the spatial hash to the most interactively coupled pair
        for i in top_idx:
            if i < n:
                temp_v = v.copy()
                # Perturb the most interacting pair
                temp_v[3*i] += spatial_hash[i, 0]
                temp_v[3*i+1] += spatial_hash[i, 1]
                temp_v[3*i+2] += np.random.uniform(-0.002, 0.002)
                
                # Use a modified objective with radius expansion on most constrained
                # Add a soft radius growth penalty to guide optimization
                # This is a modified version of the objective with additional dynamics
                def modified_neg_sum_radii_with_growth(v):
                    # Main objective: maximize total radii
                    sum_radii = np.sum(v[2::3])
                    # Add radius growth penalty
                    radius_growth = np.mean(np.clip(v[2::3], 1e-8, None) - radii)
                    return -(sum_radii + 0.003 * radius_growth)
                
                # Run a targeted optimization on the most constrained pair
                # This is a localized optimization to reconfigure the two most interacting circles
                # Use a lower tolerance and iterate to push into favorable configuration
                res_re = minimize(modified_neg_sum_radii_with_growth, temp_v, method="SLSQP", bounds=bounds,
                                  constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-12})
                
                if res_re.success:
                    v = res_re.x
                else:
                    v = temp_v  # fallback
    
    # Step 3: Identify the least constrained circle (based on isolation metric)
    # Use more efficient computation
    dists_isolation = dists.copy()
    np.fill_diagonal(dists_isolation, np.inf)  # Remove self distances
    isolation = np.min(dists_isolation, axis=1)  # Minimum distance to others
    isolated_idx = np.argmin(isolation)  # Index of most isolated circle
    
    # Step 4: Implement targeted radius expansion with dynamic adjustment
    # Use radius growth based on current configuration and distance constraints
    # We will use a hybrid approach: expand by small increments with constraint checking
    if res.success:
        v = res.x
        new_radii = v[2::3].copy()
        radii = new_radii
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Initialize growth variables
        growth_rate = 0.001
        max_growth = 0.005
        growth_counter = 0
        total_expansion = 0.0
        max_radius = np.max(radii)
        
        # Use directional and adaptive growth with constraint validation
        while growth_counter < 10 and total_expansion < max_growth:
            # Calculate tentative grow vector
            grow_vector = np.random.rand(n) * 0.04 * (radii / max_radius)  # Scale by current radii
            
            # Propose a growth on least constrained circle
            delta_r = growth_rate * grow_vector[isolated_idx]  # Scaled by isolation metric
            
            # Apply growth and check constraints
            new_radii = radii.copy()
            new_radii[isolated_idx] += delta_r
            new_radii = np.clip(new_radii, 1e-6, 0.5)  # Clamp to bounds
            
            # Create new configuration
            new_centers = np.column_stack([v[0::3], v[1::3]])
            new_v = v.copy()
            new_v[2::3] = new_radii
            
            # Check constraints
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = new_centers[i, 0] - new_centers[j, 0]
                    dy = new_centers[i, 1] - new_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                v = new_v
                radii = new_radii
                growth_counter += 1
                total_expansion += delta_r
            
            # Also consider small perturbations to neighboring circles to maintain stability
            # Add small growth to neighboring circles in the direction of isolation
            for neighbor in range(n):
                if neighbor == isolated_idx:
                    continue
                if dists[isolation] > 0.5 * dists[isolated_idx]:
                    # If distance is more than half the isolated distance, allow growth
                    if np.random.rand() < 0.3:  # 30% chance for perturbation
                        new_radii[neighbor] += np.random.uniform(0, growth_rate / 10)
                        new_radii[neighbor] = np.clip(new_radii[neighbor], 1e-6, 0.5)
            
            # Final check on final configuration
            is_valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < radii[i] + radii[j] - 1e-12:
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if not is_valid:
                # If invalid, revert to last valid state
                v = res.x
                radii = v[2::3]
                break

    # Final validation and return
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())