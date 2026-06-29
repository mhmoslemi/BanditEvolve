import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    rows = min(rows, n)  # Handle cases where rows > n (due to rounding up)
    
    # Initialize with a geometrically balanced, randomized grid with adaptive offset
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid: col-centered, row-centered
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply randomized offset + adaptive perturbation, including spatial hashing
        offset_rad = 0.025 * np.random.rand()  # Vary intensity of offset
        x = x_center + np.random.uniform(-offset_rad, offset_rad)
        y = y_center + np.random.uniform(-offset_rad, offset_rad)
        # Apply staggered grid for alternate rows
        if row % 2 == 1:
            x += 0.5 / cols
        # Apply small local spatial hashing for more diverse configurations
        spatial_hash = np.random.rand() * 0.1 * (1.0 / (1 + row))
        x += spatial_hash
        y += spatial_hash
        # Ensure not outside bounds (though should be safe due to initial placement)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3  # Conservative base radius, can be increased through expansion steps
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3 * n constraints for 3 * n variables

    def neg_sum_radii(v):
        return -np.sum(v[2::3]) 

    def get_distance_sq(i, j, v):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx*dx + dy*dy

    # Build constraints
    cons = []
    # Boundary constraints (each circle must stay inside square)
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints (each pair must have distance >= r_i + r_j)
    for i in range(n):
        for j in range(i + 1, n):
            # For performance, avoid lambda capture if possible
            # Create a unique closure per pair
            def constraint_func(i=i, j=j, v=v):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First-stage optimization: global optimization with moderate tolerance
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-10, "disp": False}
    )
    # Second-stage: targeted geometric hashing and constraint validation
    if res.success:
        v = res.x
        # Extract current configuration, including radii and positions
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        min_radius = np.min(radii)
        radius_sum = float(np.sum(radii))
        # Validate for safety
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        # Add additional geometric hashing if we're at a valid state
        if valid:
            # Apply asymmetric spatial perturbation and radius perturbation based on radii
            for i in range(n):
                # Perturb position with directional sensitivity to small radii
                direction_mag = 0.02 * (radii[i] / np.mean(radii))
                spatial_hash = np.random.rand(2) * (1 - np.cos(radii[i] * np.pi / 0.5))
                perturb_x, perturb_y = (
                    direction_mag * spatial_hash[0],
                    direction_mag * spatial_hash[1]
                )
                v[3*i] += perturb_x
                v[3*i+1] += perturb_y
                # Perturb radii with small noise for diversity in local minima
                # Only if not already close to the max limit
                if radii[i] < 0.45:
                    v[3*i+2] += 2 * (np.random.rand() - 0.5) * (radii[i] / 0.1)
                # Ensure radius remains in bounds
                v[3*i+2] = np.clip(v[3*i+2], 1e-4, 0.5)
            # Re-evaluate with new config in a new optimization pass
            res = minimize(
                neg_sum_radii,
                v,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10, "disp": False}
            )

    # Third-stage refinement: apply targeted expansion on least constrained circle with validation
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        min_radius = np.min(radii)
        radius_sum = float(np.sum(radii))
        # Validate to ensure no overlap
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        # Now, apply expansion to the least constrained circle 
        # Using a geometric optimization technique with spatial hashing
        if valid:
            # Precompute distances between all circles for finding least constrained
            dists = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dists[i, j] = np.sqrt(dx**2 + dy**2)
            
            # For each circle, find the minimal distance to any other
            min_dists = np.min(dists, axis=1)
            # Find circle with maximal min distance (i.e., least constrained)
            least_constrained_idx = np.argmax(min_dists)
            # For the least constrained circle, perform a targeted expansion with spatial hashing
            # Calculate the maximum potential expansion while preserving safety
            # Use soft constraints with perturbation for expansion
            expansion_attempts = 2
            best_attempts = {}
            best_sum = radius_sum
            best_config = v.copy()
            # We try up to expansion_attempts, applying small perturbation on each expansion
                
            for try_idx in range(expansion_attempts):
                # Perturb positions for diversity and avoid local minima
                spatial_hash = np.random.rand(n, 2) * 0.05
                perturbed_v = v.copy()
                for i in range(n):
                    if i == least_constrained_idx:
                        # Special case: we perturb the expansion circle more to allow growth
                        # This avoids the "expansion stuck" problem
                        perturbed_v[3*i] += -spatial_hash[i, 0] * 1.2
                        perturbed_v[3*i+1] += -spatial_hash[i, 1] * 0.75
                    else:
                        # Regular perturbations for other circles are minimal
                        perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
                        perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
                
                # Apply expansion to the least constrained circle only
                expand_radius = 0.005 + 0.002 * np.random.rand()  # Random expansion step
                expanded_v = perturbed_v.copy()
                expanded_v[3*least_constrained_idx + 2] = np.clip(
                    radii[least_constrained_idx] + expand_radius, 
                    1e-4, 
                    0.5
                )
                # Now validate this configuration quickly
                # Fast validation via distance check for the most constrained circles
                # Only need to check for least constrained circle's expansion effects
                new_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                valid_expanded = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = new_centers[i, 0] - new_centers[j, 0]
                        dy = new_centers[i, 1] - new_centers[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < expanded_v[3*i+2] + expanded_v[3*j+2] - 1e-12:
                            valid_expanded = False
                            break
                    if not valid_expanded:
                        break
                if valid_expanded:
                    # If valid, calculate the sum and store it
                    current_sum = np.sum(expanded_v[2::3])
                    best_attempts[try_idx] = current_sum
                    best_config = expanded_v.copy()
    
            # Use the best attempt from the expansion attempts
            v = best_config
            radius_sum = float(np.sum(v[2::3]))
            # Re-evaluate with the refined configuration
            res = minimize(
                neg_sum_radii,
                v,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-10, "disp": False}
            )
    
    v = res.x if res.success else v0  # Fallback to initial if no convergence
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())