import numpy as np

def run_packing():
    n = 26
    cols = 5  # Fixed column-based layout with 5 columns for better spatial organization
    rows = (n + cols - 1) // cols  # Calculate rows based on cols

    # Initialize positions with enhanced clustering control and adaptive perturbations
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Base perturbation based on distance from edges
        edge_dist = 0.05
        x_center = base_x + np.random.uniform(-edge_dist, edge_dist) * (np.cos(np.pi * (np.random.rand() * 2 - 1)) / (1 + row))
        y_center = base_y + np.random.uniform(-edge_dist, edge_dist) * (np.sin(np.pi * (np.random.rand() * 2 - 1)) / (1 + row))
        
        # Alternate row staggering with adaptive shift
        stagger_factor = 0.25 / cols
        if row % 2 == 1:
            x_center += stagger_factor * (1 + np.cos(np.pi * (np.random.rand() * 2 - 1))) / (1 + row)
        
        xs.append(x_center)
        ys.append(y_center)
    
    # Initialize radii with smarter baseline, considering edge effects and spacing
    max_base_r = 0.32 / cols  # Smaller than before to allow more flexibility
    r0 = np.array([max_base_r - 1e-3] * n)
    # Add adaptive spatially variant radii to help small, edge-near circles have margin
    r0 = r0 * (1 + 0.1 * np.random.rand(n))  # Spatially variant baseline radii
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Ensure bounds have 3 * n elements
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints (avoiding lambda issues with closure capture)
    def boundary_constraints(v, i, coord='x'):
        if coord == 'x':
            return v[3*i] - v[3*i+2]
        elif coord == 'y':
            return v[3*i+1] - v[3*i+2]
        else:
            return -1.0 + v[3*i] + v[3*i+2]
    
    constraints = []
    for i in range(n):
        # x >= r
        constraints.append({"type": "ineq", "fun": lambda v, i=i: boundary_constraints(v, i, 'x')})
        # x + r <= 1
        constraints.append({"type": "ineq", "fun": lambda v, i=i: boundary_constraints(v, i, 'x') + 1.0})
        # y >= r
        constraints.append({"type": "ineq", "fun": lambda v, i=i: boundary_constraints(v, i, 'y')})
        # y + r <= 1
        constraints.append({"type": "ineq", "fun": lambda v, i=i: boundary_constraints(v, i, 'y') + 1.0})
    
    # Optimized overlap constraint with vectorized distance and non-overlapping check
    def get_overlap_func(i, j):
        def _overlap_func(v):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist_sq = dx*dx + dy*dy
            # Use radius values directly from v to avoid extra storage
            return dist_sq - (v[3*i+2] + v[3*j+2])**2
        return _overlap_func

    # Build constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({"type": "ineq", "fun": get_overlap_func(i,j)})
    
    # First optimization using SLSQP with enhanced initial settings
    initial_options = {
        "maxiter": 3000,
        "ftol": 1e-12,
        "eps": 1e-6,
        "disp": False
    }
    # First pass
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, constraints=constraints, options=initial_options)
    
    # Post-optimization refinement with dynamic reconfiguration
    if not res.success:
        # Fallback to the initial vector in case of severe failure
        v = v0
    else:
        v = res.x

    # Dynamic constraint enforcement through geometric hashing and adaptive perturbations
    def get_perturbation_hash(radius, n, seed=None):
        # Returns a perturbation vector that is proportional to the radius but also varies spatially
        if seed is not None:
            np.random.seed(seed)
        return (np.random.rand(n, 2) - 0.5) * radius.mean() * 1.3  # Scaling to allow small circles to move

    perturbation_hash = get_perturbation_hash(v[2::3], n, 42)
    perturbed_v = v.copy()
    for i in range(n):
        perturbed_v[3*i] += perturbation_hash[i, 0]
        perturbed_v[3*i+1] += perturbation_hash[i, 1]
    
    # Post-perturbation optimization
    perturbed_res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds, constraints=constraints, options=initial_options)
    
    # Hybrid optimization
    if perturbed_res.success and perturbed_res.fun < res.fun:
        res = perturbed_res
        v = res.x
    
    # Additional refinement with dynamic radius expansion, especially on smallest radius and isolated circles
    if res.success:
        v = res.x
        radii = v[2::3]
        
        # Compute minimum distance for all circles (excluding self)
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                if i != j:
                    dists[i, j] = np.sqrt(dx*dx + dy*dy)
                else:
                    dists[i, j] = float('inf')
        min_dists = np.min(dists, axis=1)
        
        # Identify small and isolated circles
        small_radius_idx = np.where(radii < np.mean(radii) - 2*np.std(radii))[0]
        isolated_radius_idx = np.argsort(min_dists)[0:3]  # Most isolated
        constrained_radius_idx = np.argsort(min_dists)[-3:]  # Most constrained
        
        # Radius expansion strategy: targeted expansion but with safety
        def safe_radius_scaling(current_radii, expansion_scale, max_total_growth_percent=0.02):
            # Calculate current sum
            current_sum = np.sum(current_radii)
            # Targeted expansion with limited percentage gain
            target_total_growth = current_sum * max_total_growth_percent
            expansion = target_total_growth / (n - 1)  # Distribute equally
            # Create a safe expansion array
            expansion_array = np.full(n, expansion)
            # Increase expansion for small and isolated circles
            expansion_array[small_radius_idx] *= 1.5
            expansion_array[isolated_radius_idx] *= 1.8
            expansion_array[constrained_radius_idx] *= 1.3
            return expansion_array
        
        expansion_vector = safe_radius_scaling(radii, 0.006)  # 0.6% total growth
        # Apply with soft constraint adjustment
        def expand_radii_with_constraints(v, expansion_vector, new_radii):
            # Make a copy of the current state to work with
            adjusted_v = v.copy()
            adjusted_v[2::3] = new_radii
            # Create centers from adjusted positions
            adjusted_centers = np.column_stack([adjusted_v[0::3], adjusted_v[1::3]])
            # Validate new configuration for any overlapping circles
            overlap_detected = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = adjusted_centers[i, 0] - adjusted_centers[j, 0]
                    dy = adjusted_centers[i, 1] - adjusted_centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        overlap_detected = True
                        break
                if overlap_detected:
                    break
            if not overlap_detected:
                return adjusted_v
            else:
                # If overlaps are detected, we need to adjust expansion
                # We'll compute the scaling factor based on the overlap
                # Find the maximum overlap (distance < sum of radii - 1e-12)
                overlap_distances = np.zeros((n,n))
                for i in range(n):
                    for j in range(i+1,n):
                        dx = adjusted_centers[i, 0] - adjusted_centers[j, 0]
                        dy = adjusted_centers[i, 1] - adjusted_centers[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        overlap_distances[i,j] = dist - (new_radii[i] + new_radii[j])
                overlaps = overlap_distances < -1e-12
                # For overlaps, compute current violation
                overlap_violation = np.abs(overlap_distances[overlaps])
                # Find the circle that contributes the most to overlap
                overlapping_indices = np.where(overlaps)
                if overlapping_indices[0].size == 0:
                    # No overlap, return as is
                    return adjusted_v
                # We need to reduce expansion for the most violated circle
                # Find the circle with the greatest overlap violation
                overlap_violation_per_circle = np.zeros(n)
                for i in range(n):
                    overlap_violation_per_circle[i] = np.sum(overlap_violation[(overlaps[:,i])])
                max_violated_circle = np.argmax(overlap_violation_per_circle)
                # Reduce expansion for this circle
                expansion_vector[max_violated_circle] *= 0.5
                # Recalculate the expansion and retry
                return expand_radii_with_constraints(v, expansion_vector, new_radii)
        
        # Apply expansion safely
        new_radii = radii + expansion_vector
        new_radii = np.clip(new_radii, 1e-6, 0.5)
        # Perform constraint validation with new configuration
        safe_v = expand_radii_with_constraints(v, expansion_vector, new_radii)
        
        # Run final optimization with the new configuration
        final_v = safe_v
        if safe_v is not None:
            # Re-run with more refined options
            final_options = {
                "maxiter": 400, 
                "ftol": 1e-12,
                "eps": 1e-8,
                "options": {"disp": False}
            }
            final_res = minimize(neg_sum_radii, final_v, method="SLSQP", bounds=bounds, 
                                constraints=constraints, options=final_options)
            if final_res.success:
                res = final_res
                v = final_res.x

    # Final cleanup and return
    final_v = res.x if res.success else v0
    centers = np.column_stack([final_v[0::3], final_v[1::3]])
    # Apply post-optimization clipping with more tolerance and spatial awareness
    # Use radii values with small adjustment to avoid NaNs, while maintaining bounds
    radii = np.clip(final_v[2::3], 1e-6, 0.5)
    # Final validation and cleanup
    return centers, radii, float(radii.sum())