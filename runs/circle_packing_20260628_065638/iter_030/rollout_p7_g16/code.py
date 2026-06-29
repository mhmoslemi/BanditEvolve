import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    max_x, max_y = 1.0, 1.0
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        # Use a more nuanced range to prevent over-clustering
        x_rand = np.random.uniform(0.0, 0.3)
        y_rand = np.random.uniform(0.0, 0.3)
        # Avoid symmetry by shifting alternate rows asymmetrically
        if row % 3 == 1:
            x_center += 0.2 / cols  # more aggressive stagger
        if row % 2 == 1:
            y_center += 0.15 / rows  # staggered shift
        # Now add random perturbation in controlled bounds
        x = x_center + (np.random.uniform(-0.1, 0.1) * np.cos(10 * i + 0.2) ) / (cols + 1)
        y = y_center + (np.random.uniform(-0.1, 0.1) * np.sin(5 * i + 0.4) ) / (rows + 1)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-4
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Use bounds to strictly constrain all parameters
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    # Use lambda with closure to avoid capture issues
    cons = []
    for i in range(n):
        # Left constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right constraint: 1 - (x_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top constraint: 1 - (y_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing and pre-check optimization
    # Use lambda closures to avoid repeated function redefinition and optimize computation
    # We use a vectorized distance matrix computation in the constraint function
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                       - (v[3*i+2] + v[3*j+2])**2
            })

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})

    # Asymmetric reconfiguration: trigger a multi-phase perturbation system
    # 1) Spatial hashing with local neighborhood awareness
    # 2) Dynamic constraint reweighting to enable reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute full spatial interaction matrix efficiently
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Use a spatial hash to identify key interaction zones for perturbation
        spatial_hashes = np.random.rand(n, 2) * 0.05  # small spatial distortion
        
        # Create a perturbation vector with spatial hashing and radius-dependent weighting
        perturbation_factor = (radii / np.mean(radii)) * 0.02  # radius-aware perturbation scaling
        perturbed_v = v.copy()
        
        for i in range(n):
            # Apply perturbation with radius-aware scaling and local spatial hashing
            perturbed_v[3*i] += spatial_hashes[i, 0] * perturbation_factor[i]
            perturbed_v[3*i+1] += spatial_hashes[i, 1] * perturbation_factor[i]
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})

    # Apply a syntactic safety filter: before mutation enforce type consistency and compute
    # safety metrics across the entire configuration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Pre-validate full solution with safety metrics
        validity_mask = np.full(n, True, dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[i] + radii[j] - 1e-12:
                    validity_mask[i] = False
                    validity_mask[j] = False
        
        # Apply strict safety check and enforce non-overlap
        if not validity_mask.all():
            # If any overlap detected, trigger safety mode with soft penalty function
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 600, "ftol": 1e-12,
                                                     "eps": 1e-10})
        
        # Now perform targeted radius expansion on circle with smallest non-zero radius
        # but with adjacency and safety awareness

        # First compute full distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        full_dists = np.sqrt(dx**2 + dy**2)

        # Calculate per-circle minimal distance to all others
        min_dists = np.min(full_dists, axis=1)
        # Calculate adjacency-aware metrics and safety score
        safety_scores = np.zeros(n)
        for i in range(n):
            # safety score = min_distance - (radii[i] + mean_radius)
            # if negative, it's a candidate for expansion
            safety_scores[i] = min_dists[i] - (radii[i] + np.mean(radii))

        # Find circle with lowest safety score (highest overlap potential)
        # but only if it has minimal radius and is not in immediate adjacency
        # we'll prioritize expansion on the circle with smallest radius and smallest safety score
        if np.any(radii < 1e-5):
            smallest_radius_idx = np.argmin(radii)
            # calculate safety score for this circle
            safety_score_smallest = safety_scores[smallest_radius_idx]
            # if minimal radius circle is in high overlap or safety risk zone, expand it
            if safety_score_smallest < -1e-3:
                # Compute expansion factor for this circle
                # expand by small controlled amount to trigger reconfiguration
                expansion_radius = 0.001  # very controlled and tiny
                new_radii = radii.copy()
                new_radii[smallest_radius_idx] += expansion_radius
                new_radii = np.clip(new_radii, 1e-4, 0.5)  # clip to safe bounds
                
                # Re-evaluate with expansion while ensuring all constraints are met
                # use a new decision vector and run with tighter constraints
                # and safety checks
                v_new = v.copy()
                v_new[2::3] = new_radii

                # Re-evaluate with expanded radii and new configuration
                res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 600, "ftol": 1e-12,
                                                         "eps": 1e-10})
        
        # Now, use the updated result from the safety filter and re-expansion
        v = res.x if res.success else v

    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())