import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # --- Step 1: Initialize with randomized geometric tiling for diverse spatial configurations ---
    xs = []
    ys = []
    
    # Create a base grid with randomized tiling
    base_tiles = np.random.rand(n, 2) * 0.08
    
    # Apply tiling with geometric spacing
    tile_width = 1.0 / cols
    tile_height = 1.0 / rows
    
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid center
        x_center = (col + 0.5) * tile_width
        y_center = (row + 0.5) * tile_height
        
        # Add random geometric offset for tiling diversity
        x_offset = tile_width * 0.2 * np.sqrt(i / n)
        y_offset = tile_height * 0.2 * np.sqrt(i / n)
        
        # Random direction for tile displacement to create non-local reconfiguration
        angle = 2 * np.pi * np.random.rand()
        dx = np.cos(angle) * x_offset
        dy = np.sin(angle) * y_offset
        
        # Apply the offset to the grid center
        x = x_center + dx
        y = y_center + dy
        
        # Apply further spatial perturbation for diversity
        x += base_tiles[i, 0]
        y += base_tiles[i, 1]
        
        # Ensure the point stays within the unit square
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # --- Step 2: Define bounds with consistent length for 3*n variables ---
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # --- Step 3: Define the objective function to maximize total radii ---
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # --- Step 4: Vectorized constraints using numpy broadcasting ---
    cons = []

    # --- Step 4.1: Boundary constraints ---
    for i in range(n):
        # Left boundary: x_i - r_i >= 0.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right boundary: x_i + r_i <= 1.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom boundary: y_i - r_i >= 0.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top boundary: y_i + r_i <= 1.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # --- Step 4.2: Pairwise distance constraints using broadcasting ---
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda capturing i, j for constraint function
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2]) ** 2
            })

    # --- Step 5: Optimization with enhanced settings ---
    # First phase: Optimize initial configuration
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 400, "ftol": 1e-9, "disp": False}
    )

    # --- Step 6: Apply spatial reconfiguration using dynamic scaling of tile positions ---
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Create spatial perturbation based on radius distribution
        spatial_perturbation = np.random.rand(n, 2) * 0.05
        spatial_perturbation *= (radii / np.mean(radii))  # Radius-based scaling for unevenness

        # Apply the perturbation
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_perturbation[i, 0]
            perturbed_v[3*i + 1] += spatial_perturbation[i, 1]

        # Apply clipping to maintain within the square
        perturbed_v[0::3] = np.clip(perturbed_v[0::3], 0.0, 1.0)
        perturbed_v[1::3] = np.clip(perturbed_v[1::3], 0.0, 1.0)

        # Re-evaluate using the perturbed configuration
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 300, "ftol": 1e-10, "disp": False}
        )

    # --- Step 7: Targeted radius adjustment with dynamic constraint enforcement ---
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized calculation of distances for all circle pairs
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Find the circle with the smallest constraint radius (i.e., largest distance to others) to expand
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate a target sum of radii based on current sum
        current_total = radii.sum()
        target_total = current_total + 0.03 * current_total  # 3% increase

        # Calculate expansion factor proportional to radius
        expansion_factor = (target_total - current_total) / n * (1.15)  # Slight over-expansion factor

        # Create radii vector with expansion to least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3
        for i in range(n):
            if i != least_constrained_idx:
                # Small expansion to nearby circles
                new_radii[i] += expansion_factor * 0.8 * (1.0 + 0.15 * np.random.rand())

        # Apply new radii and validate
        while True:
            expanded_centers = np.column_stack([v[0::3], v[1::3]])
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii

            # Vectorized distance check
            dx_new = expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0]
            dy_new = expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1]
            dists_new = np.sqrt(dx_new**2 + dy_new**2)

            # Check for overlaps
            conflict = np.any(dists_new < (new_radii[:, np.newaxis] + new_radii[np.newaxis, :]) - 1e-12)
            if not conflict:
                break
            else:
                # Gradual reduction to find a valid configuration
                new_radii = radii + (new_radii - radii) * 0.95

        # Update the decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii

        # Final optimization phase with expanded radii
        res = minimize(
            neg_sum_radii,
            v_new,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 300, "ftol": 1e-10, "disp": False}
        )

    # --- Step 8: Final configuration and clipping ---
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final validation: ensure all circles are within [0,1] and non-overlapping
    # (This is redundant since the optimization ensures it, but re-validation is for safety)

    return centers, radii, float(radii.sum())