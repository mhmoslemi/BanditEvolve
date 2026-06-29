import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with geometric spacing and controlled randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols + np.random.uniform(-0.05, 0.05)
        y_center = (row + 0.5) / rows + np.random.uniform(-0.05, 0.05)
        # Staggered rows to reduce vertical clustering
        if row % 2 == 1:
            x_center += 0.4 / cols  # Reduce clustering across vertical axis
        xs.append(x_center)
        ys.append(y_center)
    # Set a slightly higher base radius for improved performance
    r0 = 0.34 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds with 3*n entries
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Object function to maximize the sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Enforce distance between centers >= sum of radii
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # First optimization with strict tolerance and high iteration
    res = minimize(neg_sum_radii, v0, 
                   method="SLSQP", bounds=bounds,
                   constraints=cons, 
                   options={"maxiter": 6000, "ftol": 1e-11, "eps": 1e-10, "disp": False})

    # First level: spatial reconfiguration with spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hashing to guide directional displacement
        spatial_hash = np.random.rand(n, 2) * 0.03
        perturbed_v = v.copy()
        for i in range(n):
            # Perturb based on spatial hash with normalization by max radius
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.max(radii)) * 0.5
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.max(radii)) * 0.5
        
        # Second level optimization with refined hash
        res = minimize(neg_sum_radii, perturbed_v, 
                       method="SLSQP", bounds=bounds,
                       constraints=cons, 
                       options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-10, "disp": False})

    # Second level: radius refinement with directed expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Identify least constrained circle (max min distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate target sum with 2.8% growth (2.0% better than 2.634)
        curr_sum = np.sum(radii)
        target_sum = curr_sum * 1.028
        # Calculate per-circle expansion factor
        expansion_factor = (target_sum - curr_sum) / n
        directional_hash = np.random.rand(n, 3) * 0.04

        # Apply directional growth
        new_radii = radii.copy()
        # First grow the least constrained circle
        new_radii[least_constrained_idx] += expansion_factor * 1.5  # Boost for least constrained
        for i in range(n):
            if i != least_constrained_idx:
                # Directional expansion with stochastic bias
                direction = directional_hash[i] * 0.6  # Normalize to prevent overgrowth
                # Compute influence based on proximity to least constrained circle
                dist_to_lesser = np.linalg.norm(centers[i] - centers[least_constrained_idx])
                if dist_to_lesser < 0.1:
                    expansion = expansion_factor * 1.6 * (1 + direction[0] * 0.5)  # Boost nearby
                else:
                    expansion = expansion_factor * 1.2 * (1 + direction[1] * 0.4)  # Base expansion
                new_radii[i] += expansion
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce by 1% if invalid
                new_radii = radii + (new_radii - radii) * 0.99
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Third level optimization with refined parameters
        res = minimize(neg_sum_radii, v_new, 
                       method="SLSQP", bounds=bounds,
                       constraints=cons, 
                       options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-10, "disp": False})

    # Final post-processing if we found a valid solution
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())