import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with advanced stochastic spatial hashing and topological perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base geometric grid with staggered rows
        row_center = (row + 0.5) / rows
        col_center = (col + 0.5) / cols
        x_center = col_center
        y_center = row_center
        
        # Apply spatial hashing to create asymmetric positioning
        spatial_hash_x = np.random.rand() * 0.07
        spatial_hash_y = np.random.rand() * 0.07
        
        # Staggered row shift with dynamic spacing
        if row % 2 == 1:
            x_center += 0.5 / cols
        
        # Apply small perturbations to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        
        xs.append(x + spatial_hash_x)
        ys.append(y + spatial_hash_y)
    
    r0 = 0.375 / cols - 1e-2
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
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Asymmetric reconfiguration: spatial hashing and topological disruption
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create asymmetric spatial hashing grid
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion with topological reordering heuristic
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle using topological awareness
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate potential expansion based on spatial distribution
        total_sum = np.sum(radii)
        potential_growth = 0.0085
        expansion_factor = potential_growth / (n - 1)
        
        # Introduce topological expansion and adjacency-aware growth
        directional_hash = np.random.rand(n, 2) * 0.04
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        
        # Expand neighbors with adjusted weight based on spatial adjacency
        for i in range(n):
            if i != least_constrained_idx:
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                expansion = expansion_factor * (1.0 + (0.05 * np.random.rand()))
                # Boost expansion for nearby and high-directional circles
                if adj_weight < 0.2:
                    expansion *= 1.5
                # Apply directional bias based on spatial hashing
                new_radii[i] += expansion * (1.0 + directional_hash[i, 0] * 0.3)
        
        # Apply expansion with constraint validation using advanced vectorized checks
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])  
            
            # Vectorized distance calculation using broadcasting with early exit
            dx_exp = expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0]
            dy_exp = expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1]
            dists_exp = np.sqrt(dx_exp**2 + dy_exp**2)
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    if dists_exp[i, j] < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Gradual reduction if expansion fails
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())