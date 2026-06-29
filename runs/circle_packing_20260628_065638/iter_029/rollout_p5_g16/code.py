import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Optimized initialization with adaptive geometric clustering and perturbation
    xs, ys = [], []
    base_grid = np.zeros((rows, cols, 2))
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            x += np.random.uniform(-0.04, 0.04)
            y += np.random.uniform(-0.04, 0.04)
            # Apply staggered offset for vertical rows
            if i % 2 == 1:
                x += (j + 0.5) / cols * 0.15
            base_grid[i, j] = [x, y]
    
    indices = np.random.permutation(n)
    for i in range(n):
        row = i // cols
        col = i % cols
        x = base_grid[row, col, 0]
        y = base_grid[row, col, 1]
        xs.append(x)
        ys.append(y)

    # Adaptive radius initialization based on grid density and spacing
    # Start with a lower bound that scales with grid spacing
    grid_spacing_x = 1.0 / cols
    grid_spacing_y = 1.0 / rows
    base_radius = 0.33 * np.min([grid_spacing_x, grid_spacing_y]) - 1e-3
    r0 = np.array([base_radius] * n)

    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Strictly enforce bounds length and decision vector consistency
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Customized objective with curvature-aware regularization
    def neg_sum_radii(v):
        """Optimize the sum of radii with smooth regularization on center density"""
        sum_radii = np.sum(v[2::3])
        
        # Add curvature-aware regularization: penalize dense clusters in the centers
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, 0, np.newaxis] - centers[np.newaxis, :, 0]
        dy = centers[:, 1, np.newaxis] - centers[np.newaxis, :, 1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Regularize local cluster densities with a soft exponential kernel
        cluster_density = np.exp(-0.5 * np.clip(dist, 0, 0.2)) * (dist > 0.01)
        cluster_density_sum = cluster_density.sum()
        cluster_density_regularizer = 0.1 * cluster_density_sum
        
        # Regularize proximity to boundaries with harmonic mean of boundary distances
        boundary_dist = np.array([
            (v[3*i] - v[3*i+2]) for i in range(n) 
            if v[3*i] - v[3*i+2] > 1e-16
        ]) + np.array([
            (1.0 - v[3*i] - v[3*i+2]) for i in range(n) 
            if (1.0 - v[3*i] - v[3*i+2]) > 1e-16
        ]) + np.array([
            (v[3*i+1] - v[3*i+2]) for i in range(n) 
            if v[3*i+1] - v[3*i+2] > 1e-16
        ]) + np.array([
            (1.0 - v[3*i+1] - v[3*i+2]) for i in range(n) 
            if (1.0 - v[3*i+1] - v[3*i+2]) > 1e-16
        ])
        if len(boundary_dist) > 0:
            boundary_regularizer = 0.1 * np.mean(1.0 / boundary_dist)
        else:
            boundary_regularizer = 0.0
        
        return -sum_radii + cluster_density_regularizer + boundary_regularizer

    # Efficient constraint generation using lambda closures with capture
    # Boundary constraints: left + radius <= 1.0
    # Right - radius >= 0.0
    # Bottom + radius <= 1.0
    # Top - radius >= 0.0
    cons = []
    for i in range(n):
        # Right constraint
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Left constraint
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Top constraint
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Bottom constraint
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Efficient overlap constraints between all pairs
    # Using vectorized lambda closures with unique captures
    for i in range(n):
        for j in range(i + 1, n):
            # Use unique lambda to avoid closure capture issues
            # This is safe since we capture i and j as local variables
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", 
                         "fun": constraint_func})

    # Initial optimization with adaptive iteration and tolerance strategy
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-10})

    # Asymmetric reconfiguration using spatial hashing with density-weighted perturbation
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute cluster density to guide perturbation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dist = np.sqrt(dx**2 + dy**2)
        dist = np.where(dist == 0, np.inf, dist)
        
        # Find local cluster centers
        cluster_density = np.zeros(n)
        for i in range(n):
            local_dists = dist[i, :]
            local_dists = local_dists[local_dists > 1e-8]
            cluster_density[i] = np.mean(1.0 / local_dists)
        
        # Generate spatial hashing with adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbation = spatial_hash * (radii / np.mean(radii)) * (1.0 + 0.1 * (cluster_density / np.max(cluster_density)))
        
        # Apply perturbation to centers
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation[i, 0]
            perturbed_v[3*i+1] += perturbation[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-10,
                                                     "maxcor": 50})
        
    # Post-reconfig optimization: targeted expansion with soft constraints
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute min distances to all others for each circle
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        
        # Find circle with maximum min distance (least constrained)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate potential expansion with conservative scaling
        current_total = np.sum(radii)
        target_total = current_total + 0.008
        
        # Calculate expansion coefficient, adjusting based on cluster density
        cluster_expansion = 1.0 + (0.005 * cluster_density[least_constrained_idx] / np.max(cluster_density))
        expansion_coefficient = (target_total - current_total) / (n - 1) * cluster_expansion
        
        # Create expansion vector
        expansion_vec = np.zeros(n)
        expansion_vec[least_constrained_idx] = expansion_coefficient * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion_vec[i] = expansion_coefficient * (1.0 + 0.1 * np.random.rand())  # Stochastic expansion
        
        # Generate expansion vector with gradient smoothing
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = radii + expansion_vec
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Fast validation using precomputed distances
            for i in range(n):
                for j in range(i+1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < (radii[i] + radii[j]) - 1e-12:
                        # Re-apply smaller expansion
                        expansion_vec = expansion_vec * 0.9
                        break
                else:
                    continue
                break
            else:
                break
        
        # Execute expansion with constraints
        v_new = v.copy()
        v_new[2::3] = radii + expansion_vec
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 250, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-10,
                                                     "maxcor": 50})
        
    # Post-expansion refinement with adaptive constraint tightening
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Tighten grid constraints for tighter packing
        for i in range(n):
            # Additional constraint tightening for boundary proximity
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i: v[3*i] - v[3*i+2] + 1e-10})
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2] - 1e-10})
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i: v[3*i+1] - v[3*i+2] + 1e-10})
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2] - 1e-10})
        
        # Final refinement with tighter constraints
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 150, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-10,
                                                     "maxcor": 50})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())