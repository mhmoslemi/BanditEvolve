import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize with a more refined spatial tiling and adaptive spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Compute central x/y based on column/row
        x_center = (col + 0.5 + np.random.normal(0, 0.1)) / cols
        y_center = (row + 0.5 + np.random.normal(0, 0.1)) / rows
        # Apply staggered alternation and slight asymmetric row offsets
        if row % 2 == 1:
            x_center += (0.5 / cols) * (np.random.normal(0, 0.2))
        xs.append(x_center)
        ys.append(y_center)

    r0 = 0.32 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensures 3*n length for 3*26

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})

    # Vectorized overlap constraints with better gradient approximation
    for i in range(n):
        for j in range(i + 1, n):
            # Use anonymous function with lambda closure for lambda captures
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # First optimization with increased iterations and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "gtol": 1e-10})
    
    # Apply radical geometric reconfiguration with directional hashing and spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create refined spatial hashing with dynamic scaling based on current spatial distribution
        spatial_hash = np.random.rand(n, 2) * 0.05
        spatial_scale = np.sqrt(np.sum((centers - np.mean(centers, axis=0))**2, axis=1)) / np.std(radii)
        
        # Apply directional spatial perturbation based on hashing, spatial scaling, and radii
        perturbed_v = v.copy()
        for i in range(n):
            # Spatial perturbation with adaptive scaling and directional influence
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii) * spatial_scale[i])
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii) * spatial_scale[i])
        
        # Re-optimize with perturbed spatial configurations
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})

    # Targeted radius expansion on least constrained circle with directional bias
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Create directional expansion with spatial hashing and dynamic expansion factor
        directional_hash = np.random.rand(n, 2) * 0.04
        expansion_factor = 0.008  # Base expansion factor
        
        # Calculate dynamic expansion based on current total and potential for expansion
        current_total = np.sum(radii)
        target_total = current_total + 0.01  # Small but non-trivial increase
        expansion_factor *= (target_total - current_total) / (n - 1)
        
        # Apply directional expansion to least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor
        
        # Apply directional expansion to adjacent circles with weighted adjacency
        for i in range(n):
            if i != least_constrained_idx:
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                if adj_weight < 0.1:
                    # Nearby circles get boosted expansion to maximize overall benefit
                    expansion = expansion_factor * 1.5
                else:
                    expansion = expansion_factor * 1.0
                new_radii[i] += expansion * (1.0 + directional_hash[i, 0] * 0.3)

        # Apply expansion with strict constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Use gradual convergence to prevent overshooting
                new_radii = radii + (new_radii - radii) * 0.97
        
        # Update decision vector with reconfigured radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-optimize with expanded configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())