import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with optimized spatial hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Asymmetric reconfiguration with spatial hashing: apply a randomized version of constraint_func
    # that introduces stochasticity in circle placement
    # First, perform initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})

    # Asymmetric reconfiguration phase
    if res.success:
        v = res.x
        # Identify the most constrained circle (least distance to other circles)
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance computation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute minimum distances for each circle
        min_dists = np.min(dists, axis=1)
        # Find the circle with the most constrained position
        most_constrained_idx = np.argmin(min_dists)

        # Generate asymmetric perturbation to reconfigure the layout
        # Introduce controlled randomness with bias towards less constrained circles
        asymmetry_factor = 1.5
        spatial_hash = np.random.rand(n, 2) * 0.03 * asymmetry_factor
        perturbed_v = v.copy()
        # Perturb only the most constrained circle for asymmetric configuration
        perturbed_v[3*most_constrained_idx] += spatial_hash[most_constrained_idx, 0]
        perturbed_v[3*most_constrained_idx+1] += spatial_hash[most_constrained_idx, 1]
        
        # Re-evaluate with perturbed spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # After asymmetric reconfiguration, apply controlled radial expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find the circle that is the least constrained in radial expansion
        # Compute how much each circle could expand while maintaining non-overlap
        possible_expand = np.zeros(n)
        for i in range(n):
            # Compute how much we could expand this circle in isolation
            # Assume all other circles remain static
            max_radius_i = 0.5
            for j in range(n):
                if j != i:
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    max_possible = dist - v[3*i+2] - v[3*j+2]
                    if max_possible < 0:
                        max_possible = 0
                    max_radius_i = min(max_radius_i, max_possible)
            possible_expand[i] = max_radius_i - v[3*i+2]
        
        # Identify the circle with the largest possible radial expansion potential
        least_constrained_radial_idx = np.argmax(possible_expand)
        
        # Targeted radius expansion on the least constrained circle
        target_total_sum = np.sum(radii) + 0.006
        # Compute how much to increase the radius of the least constrained circle
        expansion = (target_total_sum - np.sum(radii)) / (n) * 1.1
        # Set a maximum expansion boundary by checking current expansion potential
        max_expansion = possible_expand[least_constrained_radial_idx]
        expansion = min(expansion, max_expansion)
        
        # Create an expansion vector
        expanded_radii = radii.copy()
        expanded_radii[least_constrained_radial_idx] += expansion
        
        # Apply the expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = expanded_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < expanded_radii[i] + expanded_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly
                expanded_radii = radii + (expanded_radii - radii) * 0.95
        
        # Update decision vector
        v = expanded_v

    # Final optimization with refined configuration
    if res.success:
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())