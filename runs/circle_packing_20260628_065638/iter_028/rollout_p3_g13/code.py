import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Improved random initialization with enhanced grid spacing and asymmetric randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols * 1.02  # Slight expansion of grid cells
        y_center = (row + 0.5) / rows * 1.02
        # Randomized offset with asymmetric distribution for better spread
        x = x_center + np.random.uniform(-0.06, 0.04) * (1 - (2 * row / (rows - 1)))
        y = y_center + np.random.uniform(-0.04, 0.06) * (1 - (2 * row / (rows - 1)))
        # Staggered rows with dynamic offset based on row
        if row % 2 == 1:
            x += 0.5 / cols * (1 - (2 * row / (rows - 1)))
        xs.append(x)
        ys.append(y)
    
    r0 = 0.34 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Ensure bounds list has length of 3*n for decision vector
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Negative sum of radii objective
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint function with better closure handling
    cons = []
    for i in range(n):
        # Left margin constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right margin constraint: 1 - (x + r) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom margin constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top margin constraint: 1 - (y + r) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Precompute constraint pairs for all circle pairs (i, j)
    # Vectorized overlap constraint function with closure handling
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + 
                             (v[3*i+1] - v[3*j+1])**2 - 
                             (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Asymmetric reconfiguration with stochastic spatial perturbation
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Generate asymmetric spatial hash based on circle radii
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion on least constrained circle with soft constraints
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Precompute all pairwise distances in vectorized form
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate minimum distance for each circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate potential maximum expansion based on minimum distance
        min_dist = min_dists[least_constrained_idx]
        max_possible_radius = (min_dist - 1e-12) / 2  # 50% of min distance
        current_radius = radii[least_constrained_idx]
        expansion_factor = max(0.0, (max_possible_radius - current_radius) / current_radius)
        
        # Apply expansion only to least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 0.95
        
        # Re-evaluate with new configuration
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Use exact validation function from the problem statement
        def validate_expanded_config(v):
            # Extract new centers and radii
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            n = centers.shape[0]
            
            # Check positions and radii
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if (x - r < -1e-12 or x + r > 1 + 1e-12 or
                    y - r < -1e-12 or y + r > 1 + 1e-12):
                    return False, f"Circle {i} outside unit square"
                if r < 1e-4:
                    return False, "Circle has too small radius"
            
            # Check all circle overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radii[i] + radii[j] - 1e-12:
                        return False, f"Circles {i}, {j} overlap"
            
            return True, "Valid configuration"
        
        while True:
            if validate_expanded_config(expanded_v)[0]:
                break
            # If invalid, reduce expansion slightly
            new_radii = radii + (new_radii - radii) * 0.95
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii

        # Final optimization after expansion
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final configuration
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())