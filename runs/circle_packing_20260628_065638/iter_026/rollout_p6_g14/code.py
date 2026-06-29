import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize using hexagonal lattice with perturbation to avoid symmetry
    x_centers = np.linspace(0.5 / cols, 1 - 0.5 / cols, cols)
    y_centers = np.linspace(0.5 / rows, 1 - 0.5 / rows, rows)
    xs = []
    ys = []
    
    # Create staggered grid with offset for better spacing
    for row in range(rows):
        for col in range(cols):
            x = x_centers[col] + np.random.uniform(-0.04, 0.04)
            y = y_centers[row] + np.random.uniform(-0.04, 0.04)
            # Alternate row offset to simulate hexagonal tiling
            if row % 2 == 1:
                x += 0.5 / cols
            xs.append(x)
            ys.append(y)
    
    # Adjust initial radii based on cell size and grid geometry
    r0 = 0.35 / np.sqrt(cols * rows) - 1e-3
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})

    # Execute geometric transformation: spatial hashing and forced topological reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances with vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        adj = dists <= (radii + radii.reshape(-1, 1))
        
        # Build graph and find connected components for topological reordering
        from scipy.sparse import csr_matrix, csgraph
        graph = csr_matrix(adj)
        components = csgraph.connected_components(graph)[1]
        
        # Generate spatial hash for each component to trigger layout reconfiguration
        component_hash = np.random.rand(n, 2) * 0.08
        # Apply hash-based spatial perturbations to force a new layout
        for i in range(n):
            v[3*i] += component_hash[components[i], 0] * 1.2
            v[3*i+1] += component_hash[components[i], 1] * 1.2
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion on the circle with the smallest radius
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the least constrained circle (smallest radius and largest minimum distance)
        min_dists = np.min(dists, axis=1)
        # Handle edge cases: avoid division by zero and ensure valid indices
        radii[radii == 0] = 1e-6
        score = min_dists / radii
        least_constrained_idx = np.argmin(score)
        
        # Calculate target expansion and apply spatial-aware radius adjustment
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.009
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Adjust radii to ensure non-overlap and maintain feasibility
        # Apply asymmetric expansion with directional perturbation
        new_radii = radii.copy()
        direction = np.random.rand(2) - 0.5  # Random direction to perturb expansion
        direction /= np.linalg.norm(direction)  # Normalize direction vector
        
        # Expand the least constrained circle more to trigger layout change
        new_radii[least_constrained_idx] += expansion_factor * 1.3
        for i in range(n):
            if i != least_constrained_idx:
                # Apply direction-aware expansion
                new_radii[i] += expansion_factor * (0.8 + 0.2 * np.dot(direction, np.random.rand(2)))
        
        # Validate and adjust expansion if needed
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
                # Reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final optimization pass with tighter tolerances and directional perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances with vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Update minimum distances and find least constrained circle
        min_dists = np.min(dists, axis=1)
        radii[radii == 0] = 1e-6
        score = min_dists / radii
        least_constrained_idx = np.argmin(score)
        
        # Calculate expansion for final refinement
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.01
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Apply directional perturbation to expansion
        direction = np.random.rand(2) - 0.5  # Random direction for final optimization
        direction /= np.linalg.norm(direction)  # Normalize direction vector
        
        # Expand least constrained circle and apply directional expansion to others
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.1
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (0.8 + 0.2 * np.dot(direction, np.random.rand(2)))
        
        # Validate and update decision vector
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
                # Reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    # Final fallback to initial configuration if optimization fails
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())