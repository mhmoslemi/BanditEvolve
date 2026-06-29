import numpy as np

def run_packing():
    n = 26
    cols = 6  # Intentional increase to leverage spatial distribution flexibility
    rows = (n + cols - 1) // cols
    
    # Initialize with a grid + stochastic perturbation (enhanced with gradient-aware offset)
    xs_initial = np.empty(n)
    ys_initial = np.empty(n)
    r0_initial = 0.35 / cols - 1e-3
    
    # Compute grid-based base positions, enhanced with spatial-aware noise
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Use gradient-guided noise for better exploration: noise scales inversely with grid spacing
        noise_mag = 0.025 / (cols - 2) if cols > 2 else 0.025
        x_noise = np.random.normal(0, noise_mag, 1)
        y_noise = np.random.normal(0, noise_mag, 1)
        # Row-based staggering and edge avoidance
        if row % 2 == 1:
            x_center += 0.5 / cols
        # Adjust for edge cases
        x_center = np.clip(x_center + x_noise[0], 0.1, 0.9)
        y_center = np.clip(y_center + y_noise[0], 0.1, 0.9)
        xs_initial[i] = x_center
        ys_initial[i] = y_center
    
    # Initialize with more dynamic radius profile
    r0_initial = np.full(n, 0.26 / rows)  # More flexible to adapt post-optimization
    
    v0 = np.empty(3 * n)
    v0[0::3] = xs_initial
    v0[1::3] = ys_initial
    v0[2::3] = r0_initial

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * n  # Ensure 3n entries for vector of length 3n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with closure capture
    cons = []
    for i in range(n):
        # X left: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # X right: x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Y bottom: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Y top: y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})

    # Overlap constraints with dynamic distance-aware gradient handling
    # Use pairwise distance squared - (r1 + r2)^2 to avoid sqrt and for numerical stability
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with explicit i,j capture for correct indexing
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Phase 1: Baseline optimization with adaptive gradient tracking
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 1000, 
            "ftol": 1e-10, 
            "gtol": 1e-9,  # Stricter gradient tolerance
            "eps": 1e-9,   # Improve finite differences
            "disp": False
        }
    )

    # Phase 2: Spatial reconfiguration with localized stochastic perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial hash with radius-dependent perturbation to ensure more diverse exploration
        # Radius-based scaling of perturbation ensures small circles don't get disrupted
        spatial_hash = np.random.rand(n, 2) * 0.045
        perturbation_factor = np.sqrt(radii) / np.mean(np.sqrt(radii))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * perturbation_factor[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * perturbation_factor[i]
        
        # Optimize with adaptive spatial constraints
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 800,
                "ftol": 1e-11, 
                "gtol": 1e-10, 
                "eps": 5e-10,
                "disp": False
            }
        )

    # Phase 3: Asymmetric expansion on least constrained circle with gradient-aware growth
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances (vectorized)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        distances = np.sqrt(dx**2 + dy**2)
        # Compute distance to neighbors, then the minimum distance for each circle
        min_distance_per_circle = np.min(np.where(distances < 1e-12, 1e-12, distances), axis=1)
        # Compute minimum margin (distance - (r1 + r2)) for each pair
        margins = np.zeros((n, n))  # margins[i][j] = margin between i and j
        for i in range(n):
            for j in range(i+1, n):
                margins[i,j] = distances[i,j] - (radii[i] + radii[j]) 
        min_margin_per_circle = np.min(margins, axis=1)
        
        # Identify least constrained circle (maximum margin and minimal distance)
        # Weighted by 0.7 on margin, 0.3 on distance (prioritizing margin but also distance)
        constrained_weights = 0.7 * min_margin_per_circle + 0.3 * (1/min_distance_per_circle)
        least_constrained_idx = np.argmax(constrained_weights)
        
        # Compute growth potential: maximum possible growth in radius without violating constraints
        # Assume ideal expansion with minimal margin and distance constraints
        # Max possible expansion factor = min( (margins[i][j]) / (2 * radii[i]) for i in neighbors)
        # But to simplify, just use the minimal distance to neighbors as a proxy
        
        # Compute for each circle the maximum possible growth
        max_growth = np.zeros(n)
        for i in range(n):
            # Find the maximum allowable growth without overlap with neighbors
            # Assume expansion is equally distributed to neighbors if needed
            # For each neighbor j of i, compute (distance - (radii[i] + radii[j])) / (2 * (radii[i] + radii[j])) as max fraction
            # Take the minimum of these for the circle
            possible_growth = np.zeros(n)
            for j in range(n):
                if i == j:
                    continue
                # Compute growth fraction if i grows, keeping j fixed
                distance = distances[i,j]
                current_sum = radii[i] + radii[j]
                if distance < current_sum - 1e-12:
                    # Already in violation, so not growth possible
                    possible_growth[j] = -1
                else:
                    # Growth fraction = (min(distance - current_sum - 1e-12, ...)) / (2 * radii[i])
                    # Actually, it's how much can we increase r_i while keeping same r_j
                    # (d - r_i - r_j) = r_i_new - r_i -> max r_i_new = d - r_j
                    # So growth is (d - r_j - r_i) / radii[i]
                    growth_i = (distance - radii[j]) / radii[i]
                    possible_growth[j] = growth_i
            max_growth[i] = np.min(possible_growth[possible_growth != -1])
        
        # Choose the most promising expansion candidate (least constrained and max potential)
        growth_potential = max_growth * constrained_weights
        best_idx = np.argmax(growth_potential)
        
        # Calculate baseline value
        current_total = np.sum(radii)
        target_growth = 0.0035  # Conservative value for safe expansion
        total_growth = target_growth
        # Allocate growth to best_idx and others proportionally (adjust for constraint-based scaling)
        expansion = np.ones(n) * (total_growth / (n - 1))  # Assume others get equal share
        expansion[best_idx] = expansion[best_idx] * 1.8  # Enhance expansion for best candidate

        # Apply and optimize in a safe manner
        # Create expansion vector with targeted expansion on best_idx
        expanded_v = v.copy()
        expanded_v[2::3] = radii + expansion
        
        # Re-evaluate with new radii and enforce constraints in SLSQP
        res = minimize(
            neg_sum_radii,
            expanded_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 300,
                "ftol": 1e-11, 
                "gtol": 1e-10, 
                "eps": 1e-10,
                "disp": False
            }
        )

    # Final validation with adaptive fallback for stability
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # Clamp to max radius of 0.5
    
    # Final safety check: enforce non-overlap (as a safety belt, especially if optimization had issues)
    if res.success:
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[i] + radii[j] - 1e-12:
                    # If overlap, find a perturbation to fix it
                    # Simple linear push apart
                    direction = [dx, dy]
                    norm = np.sqrt(dx**2 + dy**2)
                    if norm == 0:
                        continue  # No separation, just skip
                    push_amount = (radii[i] + radii[j] - dist) * 0.98
                    direction_normalized = np.array(direction) / norm
                    centers[i] += direction_normalized * push_amount * 0.5
                    centers[j] -= direction_normalized * push_amount * 0.5
                    # Ensure after pushing, everything is within bounds
                    centers[i, 0] = np.clip(centers[i, 0], 0.0, 1.0)
                    centers[i, 1] = np.clip(centers[i, 1], 0.0, 1.0)
                    centers[j, 0] = np.clip(centers[j, 0], 0.0, 1.0)
                    centers[j, 1] = np.clip(centers[j, 1], 0.0, 1.0)
    
    return centers, radii, float(radii.sum())