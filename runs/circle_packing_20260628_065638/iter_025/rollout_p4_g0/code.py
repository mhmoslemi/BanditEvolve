import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.2, 0.2)
        y = y_center + np.random.uniform(-0.2, 0.2)
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-12})
    
    # Radical geometric reconfiguration using geometric hashing
    if res.success:
        v = res.x
        # Create a global geometric hash for spatial hashing
        geometric_hash = np.random.rand(n, 2) * 0.15
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += geometric_hash[i, 0] * 1.5
            perturbed_v[3*i+1] += geometric_hash[i, 1] * 1.5
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})

    # Implement non-local reconfiguration with spatial hashing and adjacency reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate adjacency graph and reconfigure based on distance ranking
        adj_graph = np.zeros((n, n))
        for i in range(n):
            adj_graph[i] = np.argsort(dists[i])[:int(np.sqrt(n))]  # Top sqrt(n) neighbors
        
        # Create new spatial layout with adjacency constraints
        new_centers = np.zeros((n, 2))
        for i in range(n):
            # Place first circle at random
            if i == 0:
                new_centers[i, 0] = np.random.uniform(0.2, 0.8)
                new_centers[i, 1] = np.random.uniform(0.2, 0.8)
                continue
            # Find the farthest neighbor to place the circle
            farthest_idx = np.argmax(dists[i, adj_graph[i]])
            new_center = new_centers[farthest_idx]
            # Place the new circle with distance equal to the smaller radius
            if i in adj_graph[farthest_idx]:
                dist = radii[farthest_idx] + radii[i]
            else:
                dist = 1.5 * np.min(radii)
            
            angle = np.random.uniform(0, 2 * np.pi)
            new_centers[i, 0] = new_center[0] + dist * np.cos(angle)
            new_centers[i, 1] = new_center[1] + dist * np.sin(angle)
        
        # Create new decision vector with spatial coordinates
        new_v = np.empty(3 * n)
        new_v[0::3] = new_centers[:, 0]
        new_v[1::3] = new_centers[:, 1]
        new_v[2::3] = radii
        
        # Re-evaluate with reconfigured spatial layout
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})

    # Apply controlled radius expansion on least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle with largest minimum distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Apply controlled expansion while maintaining constraints
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.006
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Expand the least constrained circle and others
        expanded_radii = radii.copy()
        expanded_radii[least_constrained_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                expanded_radii[i] += expansion_factor
        
        # Ensure no overlaps with new radii
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = expanded_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < expanded_radii[i] + expanded_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # If invalid, scale down expansion
                expanded_radii = radii + (expanded_radii - radii) * 0.98
        
        # Update decision vector
        v = expanded_v

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())