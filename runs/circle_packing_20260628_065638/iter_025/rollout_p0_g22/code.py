import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid with more variance
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering with tighter bounds
        x = x_center + np.random.uniform(-0.1, 0.1)
        y = y_center + np.random.uniform(-0.1, 0.1)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.6 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n

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
            # Use lambda closure with parameter capturing for constraint evaluation
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1500, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-8}
    )
    
    # Radical geometric hashing and spatial reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric hash with dynamic perturbation
        spatial_hash = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8}
        )

    # Targeted reordering and radius expansion on least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle with weighted minimum distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Target least constrained instead of most
        
        # Create adjacency matrix for topological reordering
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                if dists[i, j] < (radii[i] + radii[j]) * 0.8:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Topological sort to reorient adjacency
        def topological_order(adj):
            n = adj.shape[0]
            in_degree = np.sum(adj, axis=1)
            queue = np.where(in_degree == 0)[0]
            order = []
            while queue.size > 0:
                u = queue[0]
                order.append(u)
                queue = queue[1:]
                for v in range(n):
                    if adj[u, v] > 0:
                        in_degree[v] -= 1
                        if in_degree[v] == 0:
                            queue = np.append(queue, v)
            return np.array(order)
        
        order = topological_order(adj)
        new_centers = centers[order]
        new_radii = radii[order]
        
        # Apply expansion with constraint validation
        while True:
            expanded_centers = new_centers
            expanded_radii = new_radii.copy()
            
            # Expand least constrained circle by maximum allowed
            expansion = (0.008 / (n - 1)) * 1.2  # 1.2x over-target expansion
            expanded_radii[least_constrained_idx] += expansion * np.random.uniform(0.8, 1.2)
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i+1, n):
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
                # If invalid, decrease expansion slightly with random backtracking
                expansion *= 0.9
                expanded_radii[least_constrained_idx] -= expansion * np.random.uniform(0.4, 0.6)
        
        # Re-map back to original indexes
        new_centers = np.zeros((n, 2))
        new_radii = np.zeros(n)
        for i, idx in enumerate(order):
            new_centers[i] = expanded_centers[idx]
            new_radii[i] = expanded_radii[idx]
        
        # Update decision vector with new configuration
        v_new = v.copy()
        v_new[0::3] = new_centers[:, 0]
        v_new[1::3] = new_centers[:, 1]
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(
            neg_sum_radii, 
            v_new, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-8}
        )

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())