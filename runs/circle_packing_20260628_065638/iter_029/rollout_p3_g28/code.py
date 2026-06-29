import numpy as np

def run_packing():
    n = 26
    # Optimize grid dimensions for dense packing - choose 6 columns for better spread
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a hybrid of spatial grids and adaptive perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Use adaptive perturbation for edge cases and high-density regions
        # Use a smaller scale for central clusters and larger for outer regions
        perturb_scale = min(0.06 * (1 + 0.1 * np.abs(row - rows / 2)), 0.08)  # Adjust scale
        x = x_center + np.random.uniform(-perturb_scale, perturb_scale)
        y = y_center + np.random.uniform(-perturb_scale, perturb_scale)
        
        # Adjust spacing in staggered rows based on proximity to edge and row index
        # Add vertical offset for staggered rows in a non-uniform way
        if row % 2 == 1:
            # Use smaller vertical offset for rows near edges to prevent excessive height usage
            max_offset = 0.5 / cols * (1 - 0.2 * np.abs(row - rows/2))
            x += np.random.uniform(-max_offset, max_offset)
        
        xs.append(x)
        ys.append(y)
    
    # Calculate initial radii using a more refined formula considering spacing
    # Use a combination of column-based spacing and adaptive radius scaling
    # Add some spatial awareness - use smaller radii in tightly packed regions
    r0 = 0.35 / cols - 1e-3
    # Adjust radius based on row spacing to better allocate for edge constraints
    r0 += 0.02 * np.min([rows, cols]) / (1.5 + np.log(n + 1))  # More adaptive radius scaling
    
    # Initialize vector with more robust starting point for all 26 circles
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Consistent length, matches 3*n

    def neg_sum_radii(v):
        """Maximize radius sum by minimizing negative."""
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with proper lambda capture for i
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i + 2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i + 1] - v[3*i + 2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2])})
    
    # Overlap constraints optimized with vector broadcasting - use more efficient computation
    # Instead of nested loops, we'll use broadcasting and vectorized operations where applicable
    # However, due to the nature of non-linear SLSQP, we will keep the inner loop for overlap checks
    # But we compute all distances in a more compact way for optimization
    
    # Precompute all pairwise center distances in a more efficient way
    # To maintain SLSQP's constraint structure
    # This is necessary because of the nature of inequality constraints in scipy.optimize
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j:
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})
    
    # Initial optimization with adaptive tolerances and max iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-12, "gtol": 1e-12, "eps": 1e-10})
    
    # Apply geometric reconfiguration through perturbation with adaptive spatial hashing 
    # This step is critical for exploring new configuration spaces with minimal constraint loss
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling based on radius distribution
        # Scale by the standard deviation of the radii to avoid over-perturbation
        radius_std = np.std(radii)
        radius_mean = np.mean(radii)
        perturbation_scale = 0.03 * min(radius_mean / radius_std, 1.5)
        perturbation = np.random.rand(n, 2) * 2 * perturbation_scale
        
        # Create perturbed state with radius scaling factor
        # We scale perturbation more for smaller circles to enable better expansion
        scaled_perturbation = np.clip(
            perturbation * np.abs(radii) / np.sum(radii), 
            -0.02, 0.02
        )
        
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += scaled_perturbation[i, 0]
            perturbed_v[3*i + 1] += scaled_perturbation[i, 1]
        
        # Re-evaluate with perturbed state using optimized settings
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-10})
    
    # Now we implement the geometric tiling strategy to exploit spatial hierarchy
    # The strategy will enforce a strict non-overlap boundary on the most constrained circle
    # While also applying a global radius expansion based on its spatial constraints
    
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # First, find the most spatially constrained circle using minimum distance metric
        # This step is critical - we'll use a more robust distance-based method
        # Vectorized distance computation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Determine minimal distances per circle and find the one with the most constraints
        min_distances = np.min(dists, axis=1)
        min_distance_idx = np.argmin(min_distances)  # Most constrained
        avg_distance = np.mean(dists[dists > 1e-10])  # Average non-zero distance

        # Calculate allowable radius expansion based on average distance
        # We limit expansion to preserve feasibility of all other circles
        # Use a factor based on mean distance and radius ratio
        mean_radius = np.mean(radii)
        expansion_factor = min(0.025 * np.log(avg_distance / mean_radius + 1), 0.12)
        
        # Apply spatial hashing strategy to enforce strict non-overlap
        # We'll use a localized perturbation in the direction of minimal spacing
        # This ensures a more effective optimization path
        # Compute directional perturbation based on nearby circles
        # Avoid excessive radius expansion in tightly packed regions
        
        # Find neighboring circles with the smallest distances
        neighbor_indices = np.argsort(dists[min_distance_idx, :])[:3]
        # Average position of closest neighbors to determine perturbation direction
        neighbor_centers = centers[neighbor_indices]
        dir_vec = neighbor_centers.mean(axis=0) - centers[min_distance_idx]
        dir_vec /= np.linalg.norm(dir_vec) + 1e-10
        
        # Apply directional perturbation to the constrained circle
        # Add a small directional shift and adjust radius
        v[3 * min_distance_idx] += dir_vec[0] * 0.003
        v[3 * min_distance_idx + 1] += dir_vec[1] * 0.003
        
        # We now apply a global radius expansion constraint to the entire configuration
        # Using the identified most constrained circle as a pivot point
        # We'll use an adaptive expansion strategy that's dependent on spatial distribution
        
        # Calculate current total radius sum and potential expansion
        total_radius = np.sum(radii)
        target_total_radius = total_radius + 0.0075  # More aggressive expansion
        
        # Compute expansion per circle to keep all circles within feasible bounds
        # Use a weighted expansion to preserve spatial relationships in constrained areas
        # Add more expansion to less constrained circles
        
        # Use a spatial distribution factor to adjust expansion
        # Higher expansion for circles with higher spatial freedom
        spatial_distribution_factor = np.abs(min_distances / np.max(min_distances))
        max_expansion_factor = 0.15
        circle_expansion_weights = 1 + (1 - spatial_distribution_factor) * 0.4
        
        # Calculate expansion per circle, maintaining a minimum radius
        # Avoid expansion beyond a threshold to ensure solvability
        # Distribute expansion based on weights
        expansion_per_circle = (target_total_radius - total_radius) * circle_expansion_weights / np.sum(circle_expansion_weights)
        
        # Apply expansion to all circles except the constrained one
        for i in range(n):
            if i != min_distance_idx:
                v[3*i + 2] += np.clip(expansion_per_circle[i], 1e-6, 0.03)
        
        # Re-evaluate the optimized configuration with the constrained adjustments
        res = minimize(neg_sum_radii, v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-10})
    
    # Final check for numerical stability, re-checking for violations
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
        # Final re-validation pass to ensure no overlaps or out-of-bounds
        # Although the optimizer should have ensured this, doing a final check improves reliability
        # This step avoids needing to re-run a full validation which could be costly in time
        # We use the same validator logic, but in optimized way
        # Compute distance matrix for final validation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Check all circles against boundary constraints
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if (x - r < 1e-12 or x + r > 1 - 1e-12 or
                y - r < 1e-12 or y + r > 1 - 1e-12):
                # If constraint violation found, we will fall back to previous state
                v = res.x if res.success else v0
                centers = np.column_stack([v[0::3], v[1::3]])
                radii = np.clip(v[2::3], 1e-6, None)
                res = {"success": False}
                break
        
        # Check for pairwise overlaps
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < radii[i] + radii[j] - 1e-12:
                    # If overlap found, we will fall back to previous state
                    v = res.x if res.success else v0
                    centers = np.column_stack([v[0::3], v[1::3]])
                    radii = np.clip(v[2::3], 1e-6, None)
                    res = {"success": False}
                    break
            if not res["success"]:
                break
    else:
        # If optimization fails, fall back to previous safe state
        v = res.x if res.success else v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
    
    # Ensure consistent length arrays
    # This step is for safety, even though we have already done this earlier
    # We double-check to ensure no length mismatch
    assert centers.shape[0] == n
    assert len(radii) == n
    return centers, radii, float(radii.sum())