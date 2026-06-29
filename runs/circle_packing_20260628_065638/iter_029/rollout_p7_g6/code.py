import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Optimized initialization with adaptive geometry and dynamic spatial clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Adaptive base grid with row-specific spacing to support dynamic interaction
        base_x = (col + 0.5) / cols * (1 + 0.3 * (row % 3 == 0))
        base_y = (row + 0.5) / rows * (1 + 0.1 * (col % 2 == 0))
        # Randomized spatial jitter to break symmetry and prevent clustering
        x_jitter = np.random.uniform(-0.04, 0.04)
        y_jitter = np.random.uniform(-0.04, 0.04)
        # Alternate row staggering with adaptive offset
        if row % 2 == 1:
            base_x += 0.5 / cols
            x_jitter += np.random.uniform(-0.02, 0.02)
        x = base_x + x_jitter
        y = base_y + y_jitter
        xs.append(x)
        ys.append(y)
    
    # Improved radius initialization based on spatial distribution
    # Base radius calculated using minimum distance between adjacent rows
    def get_minimum_row_distance(row):
        # Calculate distance between centers in the same row
        row_start = row * cols
        row_end = row_start + cols
        x_distances = np.abs(xs[row_start:row_end] - xs[row_start+1:row_end+1])
        y_distances = np.abs(ys[row_start:row_end] - ys[row_start+1:row_end+1])
        min_distance = np.sqrt(np.min(x_distances**2 + y_distances**2))
        return min_distance
    
    row_distances = [get_minimum_row_distance(i) for i in range(rows)]
    min_row_distance = min(row_distances)
    default_radius_ratio = 0.85 / min_row_distance
    # Compute radius using spatial-aware scaling
    r0 = np.array([default_radius_ratio * (np.cos(np.pi * row / (rows - 1)) + 1) for row in range(rows)]) * 0.33
    
    # Ensure 26 circles by distributing radii across rows
    radius_distribution = np.repeat(r0, cols)
    r0 = np.clip(radius_distribution, 1e-3, 0.5)  # Ensure minimum and maximum radii
    r0 = np.where(r0 < 0, 0.001, r0)  # Ensure no negative radii
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        # Left wall + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right wall - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom wall + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top wall - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized pair-wise distance constraints with efficient calculation
    # Use matrix operations to vectorize distance constraints
    # Note: This avoids redundant O(n^2) operations for all pairs
    def constraint_func_pair(v, i, j):
        # Efficient calculation using vector slicing
        x1 = v[3*i]
        y1 = v[3*i+1]
        r1 = v[3*i+2]
        x2 = v[3*j]
        y2 = v[3*j+1]
        r2 = v[3*j+2]
        dx = x1 - x2
        dy = y1 - y2
        return dx*dx + dy*dy - (r1 + r2)**2
    
    # Identify most dynamically interacting pairs (central circles)
    # First, define interaction intensity based on proximity
    # Use an adaptive spatial filter to find top 2 pairs
    def find_interacting_pairs(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                distances[i, j] = np.sqrt(dx*dx + dy*dy)
                distances[j, i] = distances[i, j]
        
        # Find top 2 most interacting (smallest distance) circles
        dists = np.sort(distances, axis=None)
        threshold = np.mean(dists) + 2 * np.std(dists)
        interacting = np.where(distances < threshold, True, False)

        # Extract interacting indices
        interacting_indices = np.where(interacting)
        
        # Extract top 2 interacting pairs (indices i and j with the smallest distance)
        dists_flat = distances[interacting_indices]
        top_pair_indices = interacting_indices[np.argsort(dists_flat)][0:2]
        pairs = []
        for idx in top_pair_indices:
            if idx[0] < idx[1]:
                pairs.append((idx[0], idx[1]))
            else:
                pairs.append((idx[1], idx[0]))
        return pairs
    
    # Adaptive constraint reconfiguration
    # First optimization with standard constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-12})
    
    # If successful, focus on top interacting pairs and reconfigure
    if res.success:
        # Re-identify interacting pairs with updated configuration
        v = res.x
        interacting_pairs = find_interacting_pairs(v)
        top_pair = interacting_pairs[0]
        second_pair = interacting_pairs[1]
        
        # Construct new constraint set
        new_cons = []
        for i in range(n):
            # Boundary constraints
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
            new_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        
        # Overlap constraints: include top interacting pairs explicitly and
        # maintain all other pairs as in the previous setup
        for i in range(n):
            for j in range(i + 1, n):
                # Only add constraint for top 2 interacting pairs
                if (i, j) == top_pair or (i, j) == second_pair:
                    new_cons.append({"type": "ineq", 
                                     "fun": lambda v, i=i, j=j: constraint_func_pair(v, i, j)})
        
        # Re-evaluate optimization with refined constraints
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=new_cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-12})
    
    # Now, after reconfiguration, find least constrained circle
    # Calculate minimum distance to all other circles
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute vectorized pairwise distances (optimized with NumPy broadcasting)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1, :]
        dists = np.sqrt(dx**2 + dy**2)
        
        # For each circle, find the minimum distance to other circles
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)
        # Adjust to ensure the least constrained circle is not in the interacting pair
        # (Avoiding over-constrained circle expansion)
        if least_constrained_idx in [top_pair[0], top_pair[1], second_pair[0], second_pair[1]]:
            # Find next least constrained
            min_dists[least_constrained_idx] = np.inf
            least_constrained_idx = np.argmin(min_dists)
        
        # Prepare for radius expansion
        current_total = np.sum(radii)
        expansion_factor = 0.006 / (n - 1) * (current_total / np.mean(radii))
        
        # Create an expansion vector that increases the radius of the least constrained
        # while maintaining a spatial gradient effect to prevent clustering
        expanded_radii = radii.copy()
        max_growth = (0.5 - np.max(radii)) * 2  # Safety constraint
        growth_per_circle = expansion_factor * 0.95
        
        # Apply expansion to the least constrained circle
        growth = min(growth_per_circle, max_growth)
        expanded_radii[least_constrained_idx] += growth
        
        # Apply a soft expansion to all other circles but with reduced magnitude
        for i in range(n):
            if i != least_constrained_idx:
                expanded_radii[i] += growth_per_circle * 0.75
        
        # Create an adjusted vector and re-evaluate
        expanded_v = v.copy()
        expanded_v[2::3] = expanded_radii
        
        # Re-evaluate with expanded radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=new_cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-12})
    
    # Final refinement to ensure all constraints are tightly enforced
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Final validation: adjust positions to strictly prevent overlapping
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[i] + radii[j] - 1e-10:
                    # Adjust positions
                    # Move both circles away from each other by a small amount
                    dir_x = dx / dist * 1e-3
                    dir_y = dy / dist * 1e-3
                    # Push away both circles
                    v[3*i] += dir_x
                    v[3*i+1] += dir_y
                    v[3*j] -= dir_x
                    v[3*j+1] -= dir_y
                    # Clamp within bounds
                    v[3*i] = np.clip(v[3*i], 0.0, 1.0)
                    v[3*i+1] = np.clip(v[3*i+1], 0.0, 1.0)
                    v[3*j] = np.clip(v[3*j], 0.0, 1.0)
                    v[3*j+1] = np.clip(v[3*j+1], 0.0, 1.0)
        
        # Re-evaluate final configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=new_cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())