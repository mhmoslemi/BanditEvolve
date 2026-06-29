import numpy as np

def run_packing():
    n = 26
    # Optimize tiling structure to balance space and density, with adaptive geometry
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    # Define adaptive spatial hashing to break symmetry and explore higher dimensional configuration space
    spatial_hash_radius = 0.12
    spatial_hash_scale = 0.14  # Scale for relative displacement
    
    # Initial randomized cluster centers with geometric symmetry breaking and spatial hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Add geometric hashing to cluster centers
        # This uses a hash function to displace points spatially for exploration
        hash_x = np.random.rand(n) * spatial_hash_radius
        hash_y = np.random.rand(n) * spatial_hash_radius
        
        # Apply asymmetric displacement based on grid parity for enhanced structure
        if row % 3 == 0:  # Even rows get slight vertical displacement
            base_y += 0.04
        if col % 3 == 0:  # Even columns get slight horizontal displacement
            base_x += 0.04
        
        # Create spatial hash-based offset for each circle
        x = base_x + hash_x[i] * np.sin(i * np.pi / 6.0)
        y = base_y + hash_y[i] * np.cos(i * np.pi / 6.0)
        # Add grid-specific vertical staggering, especially on alternate columns
        if col % 2 == 1:
            y += 0.025
        xs.append(x)
        ys.append(y)
    
    # Start with a reasonable initial radius based on spacing
    base_radius_start = 0.25 / cols
    r0 = base_radius_start + 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure bounds list and vector are of length 3n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-3, 0.5)]  # Slightly increased lower radius bound

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Construct constraints with lambda captures for performance
    cons = []
    for i in range(n):
        # Left margin constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right margin constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom margin constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top margin constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Construct pairwise distance constraints with lambda closures
    for i in range(n):
        for j in range(i + 1, n):
            # Avoid using global variables, use lambda with captured i and j
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                    - (v[3*i+2] + v[3*j+2])**2
            })

    # First optimization with aggressive settings to find local maxima
    res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1600, "ftol": 1e-10, "gtol": 1e-9, "disp": False}
    )
    
    # First post-optimization reconfiguration: spatial hashing to break symmetry
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Generate perturbation with geometric hashing based on positions
        spatial_hash = np.random.rand(n, 2)
        # Introduce asymmetric bias for horizontal and vertical shifts
        horizontal_shift = spatial_hash[:, 0] * (radii / np.mean(radii)) * 0.15
        vertical_shift = spatial_hash[:, 1] * (radii / np.mean(radii)) * 0.12
        perturbation = np.column_stack([horizontal_shift, vertical_shift])

        # Apply perturbations to the center positions, maintaining bounds
        perturbed_v = v.copy()
        for i in range(n):
            # Add spatial hash displacement based on radius and position
            perturbed_v[3*i] += np.clip(perturbation[i,0], -0.1, 0.1)
            perturbed_v[3*i+1] += np.clip(perturbation[i,1], -0.1, 0.1)

        # Second optimization with reduced steps to find better local maxima
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-9, "disp": False}
        )

    # Second post-optimization reconfiguration: targeted radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Calculate current radius sum and find most loosely packed circle
        total_sum = np.sum(radii)
        # Use more aggressive expansion based on geometric analysis
        # Calculate pairwise distances for constraint-based expansion
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the circle where the minimum distance to others is maximized
        min_dists = np.min(dists, axis=1)
        # Avoid circles with zero distance
        min_dists = np.where(min_dists < 1e-8, np.inf, min_dists)
        least_constrained_idx = np.argmin(min_dists)
        
        # Calculate max possible expansion before overlap
        target_radius = radii[least_constrained_idx] * 1.3  # Aggressive expansion
        max_possible_radius = 0.5  # Physical maximum
        current_radius = radii[least_constrained_idx]
        
        # Calculate how much we can expand before hitting maximum or overlapping
        expansion_limit = min(
            (max_possible_radius - current_radius), 
            (target_radius - current_radius)
        )
        # We will attempt to expand up to target_radius or max limit
        delta_radius = np.clip(expansion_limit, 0.001, None)
        
        # Calculate how to distribute delta_radius to other circles to maintain total_sum + delta_radius
        # Create a new radius vector with expansion on the least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += delta_radius
        delta_total = delta_radius
        
        # Distribute the remaining expansion to other circles while maintaining non-overlap
        # Use an adaptive expansion factor that scales with the inverse distance to others
        # This avoids expanding overlapping circles
        for i in range(n):
            if i != least_constrained_idx:
                min_dist_i = np.min(dists[i])
                if min_dist_i < 0.1:  # If circle is close to others, limit expansion
                    # Use radius scaling to avoid overlapping
                    # Expansion factor based on inverse distance to closest circle
                    factor = max(1.0 - min_dist_i / 0.15, 0.7)  # Cap at 0.7 expansion
                    new_radii[i] = radii[i] * factor
                else:
                    new_radii[i] += delta_total * (0.2 + np.random.uniform(0.05, 0.3))
        
        # Apply the new radii with constraint validation
        while True:
            # Create new vector with updated radii
            new_v = v.copy()
            new_v[2::3] = new_radii
            
            # Recalculate centers
            new_centers = np.column_stack([new_v[0::3], new_v[1::3]])
            
            # Check for all pairwise non-overlap
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = new_centers[i, 0] - new_centers[j, 0]
                    dy = new_centers[i, 1] - new_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid or delta_total <= 0.001:  # Prevent infinite loop on edge
                break
            else:
                # If invalid, reduce expansion
                delta_total *= 0.8
                
        # Update the decision vector
        v = new_v
        radii = v[2::3]
        
        # Final optimization after radius expansion
        res = minimize(
            neg_sum_radii, 
            v, 
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-9, "disp": False}
        )
    
    # Final adjustment for edge cases and normalization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        for i in range(n):
            if centers[i, 0] - radii[i] < -1e-10:
                v[3*i] = radii[i]
            elif centers[i, 0] + radii[i] > 1.0 + 1e-10:
                v[3*i] = 1.0 - radii[i]
            if centers[i, 1] - radii[i] < -1e-10:
                v[3*i+1] = radii[i]
            elif centers[i, 1] + radii[i] > 1.0 + 1e-10:
                v[3*i+1] = 1.0 - radii[i]
        
        # One final check for boundary conditions
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if (x - r < -1e-12 or x + r > 1 + 1e-12 or
                y - r < -1e-12 or y + r > 1 + 1e-12):
                v = res.x
                break  # Don't overwrite valid result if already good
        
        # Final clipping to handle numerical instability
        radii = np.clip(v[2::3], 1e-6, 0.5)
        centers = np.column_stack([v[0::3], v[1::3]])
    
    # Safe fallback if optimization fails
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())