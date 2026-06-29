import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Enhanced grid setup with adaptive spacing and hierarchical initialization
    # 1. Initial grid with more uniform spacing to avoid early clustering
    grid_centers = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Base grid: 1.2x spacing to allow for radial expansion
        base_x += (col % 2) * 0.01
        base_y += (row % 2) * 0.01
        
        # Hierarchical randomization: 
        # 1. Base perturbation based on relative position
        # 2. Dynamic stochastic perturbation to reduce initial symmetry
        # 3. Staggering to create layered structure
        x_offset = np.random.uniform(-0.06, 0.06)
        y_offset = np.random.uniform(-0.06, 0.06)
        if row % 3 == 0:
            x_offset += np.random.uniform(-0.02, 0.02)
        if row % 3 == 1:
            y_offset += np.random.uniform(-0.02, 0.02)
        
        # Row-wise staggering with relative scaling
        x = base_x + x_offset
        y = base_y + y_offset
        if row % 2 == 1:
            x += (np.random.uniform(-0.03, 0.03) if col < cols//2 else 
                  np.random.uniform(0.03, 0.06))
        if col % 2 == 0:
            y += np.random.uniform(-0.03, 0.03)
        
        grid_centers.append( (x, y) )
    
    # Radius initialization: 
    # 1. Calculate base radius using adaptive spacing 
    # 2. Ensure initial layout doesn't force early constraints
    # 3. Start with slightly lower base for room for expansion
    # Base spacing estimate using grid spacing to allow expansion
    base_spacing = np.sqrt( (grid_centers[1][0]-grid_centers[0][0])**2 + 
                            (grid_centers[1][1]-grid_centers[0][1])**2 )
    min_radius = 0.25 / (cols * np.sqrt(2)) # More conservative than previous
    # Add slight expansion potential for asymmetric growth
    r0 = min_radius - 1e-3
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array([x for x, y in grid_centers])
    v0[1::3] = np.array([y for x, y in grid_centers])
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Optimized constraints: use vectorized expressions with closure capturing
    cons = []
    for i in range(n):
        # X boundary constraints: x - radius >= 0, 1 - x - radius >=0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Y boundary constraints: y - radius >=0, 1 - y - radius >=0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Optimized pairwise constraints using vectorized distance calculation
    # Avoid full matrix computation, apply sparse constraint evaluation
    # To reduce computation and ensure gradient consistency
    for i in range(n):
        for j in range(i+1, n):
            # Create lambda that captures i and j to avoid clashing closures
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                         - (v[3*i+2] + v[3*j+2])**2})

    # Primary optimization with adaptive step control
    # Use tighter tolerances and more steps for better initial configuration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, 
                                             "gtol": 1e-10, "eps": 1e-12})

    if not res.success:
        # Fallback perturbation with hierarchical adjustment
        # Perturb with adaptive scaling based on radii
        v = v0
        radii = v[2::3]
        perturbation = np.random.rand(n, 2) * 0.06
        adjusted_perturbation = perturbation * (radii / np.clip(np.mean(radii), 1e-6, None))
        for i in range(n):
            v[3*i] += adjusted_perturbation[i, 0]
            v[3*i+1] += adjusted_perturbation[i, 1]
            # Ensure stays within bounds for x and y
            v[3*i] = np.clip(v[3*i], 0.0, 1.0)
            v[3*i+1] = np.clip(v[3*i+1], 0.0, 1.0)
        # Retry with adjusted starting point
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 3000, "ftol": 1e-12})

    if res.success and np.any(res.x[2::3] < 1e-6):
        # Ensure all radii are above minimal threshold
        v = res.x.copy()
        min_radius = 1e-6
        radii = v[2::3]
        for i in range(n):
            if radii[i] < min_radius:
                # Distribute the minimal radius to the most isolated circle
                v[3*i + 2] = min_radius
                # Adjust other circles by proportionate redistribution
                # Maintain total sum as a constraint
                total_sum = sum(v[2::3])
                delta = total_sum - 26 * min_radius
                if delta > 0:
                    # Distribute extra to least constrained circle
                    dists = np.sqrt( (v[3*i::3] - v[3*j::3])**2 + 
                                     (v[3*i+1::3] - v[3*j+1::3])**2 
                                   for i in range(n) for j in range(n) if i != j
                                   )
                    # Re-compute distances
                    dists = np.zeros((n, n))
                    for i in range(n):
                        for j in range(n):
                            dx = v[3*i] - v[3*j]
                            dy = v[3*i+1] - v[3*j+1]
                            dists[i, j] = np.sqrt(dx**2 + dy**2)
                    # Find most isolated circle
                    isolation = np.min(dists, axis=1)
                    isolated_idx = np.argmax(isolation)
                    # Add remaining delta to this circle
                    v[3*isolated_idx + 2] += delta
                    # Final adjustment
                    v = res.x.copy()
                    v[2::3] = np.clip(v[2::3], min_radius, None)
                    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                                   constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
                else:
                    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                                   constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
    
    # Asymmetric reconfiguration with dynamic spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Generate adaptive hash based on relative positions and radii
        # This creates spatial "anchors" that enable asymmetric movement
        # Use a modified Voronoi structure for dynamic hashing
        # Compute Voronoi diagram to find spatial anchor points
        from scipy.spatial import Voronoi
        # Create Voronoi diagram of current configuration
        # Voronoi diagram is computed on center positions
        vor = Voronoi(centers)
        
        # Extract Voronoi vertices and region indices
        # This provides spatial anchors that allow more informed perturbation
        hash_map = np.random.rand(n, 2) * 0.02
        # Scale perturbation based on radii
        scale_factor = np.sqrt(np.sum(radii**2)) / (n * np.mean(radii)) 
        perturbed_v = v.copy()
        
        for i in range(n):
            # Compute spatial displacement based on Voronoi structure
            # Use region information to influence perturbation
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            # Ensure region is valid (not -1)
            if region != -1 and len(region) > 3:
                # Pick a vertex in the region as spatial influence point
                vertex_idx = np.random.choice(len(region))
                v_x, v_y = vor.vertices[region[vertex_idx]]
                # Compute vector from current to influence point
                dir_x = v_x - centers[i, 0]
                dir_y = v_y - centers[i, 1]
                # Normalize and scale by radius to ensure controlled movement
                dir_unit = np.sqrt(dir_x**2 + dir_y**2)
                if dir_unit > 0:
                    dir_x /= dir_unit
                    dir_y /= dir_unit
                    perturbation_x = dir_x * (radii[i] * scale_factor)
                    perturbation_y = dir_y * (radii[i] * scale_factor)
                else:
                    perturbation_x = np.random.uniform(-0.01, 0.01)
                    perturbation_y = np.random.uniform(-0.01, 0.01)
                perturbed_v[3*i] += perturbation_x
                perturbed_v[3*i+1] += perturbation_y
            else:
                # Default perturbation if region is invalid
                perturbed_v[3*i] += np.random.uniform(-0.01, 0.01)
                perturbed_v[3*i+1] += np.random.uniform(-0.01, 0.01)
        
        # Apply clipping to maintain unit square bounds
        for i in range(n):
            perturbed_v[3*i] = np.clip(perturbed_v[3*i], 0.0, 1.0)
            perturbed_v[3*i+1] = np.clip(perturbed_v[3*i+1], 0.0, 1.0)
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, 
                                                 "eps": 1e-12})

    # Secondary optimization with targeted growth
    # Identify least constrained circle via improved isolation metric
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance computation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute isolation score
        # Improved isolation metric: 
        # (1) Mean distance to others
        # (2) Min distance to boundary (considering radius)
        # (3) Radius size
        # Combined metric for least constrained circle
        mean_dists = np.mean(dists, axis=1)
        min_dist_to_boundary = np.min( [np.min(centers[:,0] - radii), 
                                       np.min( np.abs(centers[:,0] - 1.0) - radii ),
                                       np.min( centers[:,1] - radii ),
                                       np.min( np.abs(centers[:,1] - 1.0) - radii )], axis=0)
        isolation_scores = (mean_dists + min_dist_to_boundary) * (1.0 / (radii + 1e-6)) 
        isolates_idx = np.argmin(isolation_scores)
        
        # Apply adaptive expansion 
        # 1. Expand the most isolated circle by small amount
        # 2. Allow for dynamic reconfiguration by perturbation
        # 3. Use soft expansion to ensure feasibility
        total_current_sum = np.sum(radii)
        # Calculate expansion based on isolation score and relative position
        expansion_factor = np.min([0.008, (isolation_scores[isolates_idx] / np.max(isolation_scores)) * 0.01])
        # Perturb spatial position to allow expansion
        # Use Voronoi-based spatial adjustment again
        vor = Voronoi(centers)
        
        # Check if isolates_idx is valid
        if isolates_idx < n and isolates_idx >=0:
            region_idx = vor.point_region[isolates_idx]
            region = vor.regions[region_idx]
            # Ensure region is valid (not -1)
            if region != -1 and len(region) > 3:
                # Pick a vertex in the region as spatial influence point
                vertex_idx = np.random.choice(len(region))
                v_x, v_y = vor.vertices[region[vertex_idx]]
                # Compute vector from current to influence point
                dir_x = v_x - centers[isolates_idx, 0]
                dir_y = v_y - centers[isolates_idx, 1]
                # Normalize and scale by radius to ensure controlled movement
                dir_unit = np.sqrt(dir_x**2 + dir_y**2)
                if dir_unit > 0:
                    dir_x /= dir_unit
                    dir_y /= dir_unit
                    perturbation_x = dir_x * (radii[isolates_idx] * 0.1)
                    perturbation_y = dir_y * (radii[isolates_idx] * 0.1)
                else:
                    perturbation_x = np.random.uniform(-0.005, 0.005)
                    perturbation_y = np.random.uniform(-0.005, 0.005)
                v[3*isolates_idx] += perturbation_x
                v[3*isolates_idx+1] += perturbation_y
                # Clamp to unit square
                v[3*isolates_idx] = np.clip(v[3*isolates_idx], 0.0, 1.0)
                v[3*isolates_idx+1] = np.clip(v[3*isolates_idx+1], 0.0, 1.0)
                # Expand radius by calculated amount
                v[3*isolates_idx+2] += expansion_factor
                # Ensure radius doesn't exceed max
                v[3*isolates_idx+2] = np.clip(v[3*isolates_idx+2], 1e-6, 0.5)
            
            # Re-evaluate with adjusted configuration
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
        
        if not res.success:
            # Fallback reconfiguration to fix invalid layout if needed
            v = res.x
            # If expansion was not successful, try reducing the expansion factor
            expansion_factor = max(0.004, expansion_factor * 0.6)
            v[3*isolates_idx+2] += expansion_factor
            v[3*isolates_idx+2] = np.clip(v[3*isolates_idx+2], 1e-6, 0.5)
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())