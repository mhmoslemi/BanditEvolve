import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    if cols < 5:
        cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with staggered grid and randomized offsets
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on grid spacing and a slightly more aggressive size
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds with correct length for the vector
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Set up spatial constraints with explicit closure capturing
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

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with proper i,j closure
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})

    # Reconfiguration with asymmetric spatial perturbation
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate distances from each circle to all others
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Identify least constrained circle: one with maximum minimum distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Perturbation with weighted spatial hashing
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with perturbed spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted expansion with soft constraints and gradient-based scaling
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Recalculate distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Identify best candidate for expansion
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate current total sum and growth target
        current_total = np.sum(radii)
        expansion_target = 0.008  # Slight increase above previous best
        
        # Compute expansion factor based on least constrained radius and current state
        expansion = expansion_target / (n - 1) * (current_total / np.sum(radii))
        expansion += 0.001 * np.random.rand()  # Small stochastic perturbation
        
        # Create expanded radii vector
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion * 1.3  # Boost for best candidate
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion * 0.95 * (1 + 0.1 * np.random.rand()) # Gradual spread
        
        # Validate and adjust radii to maintain non-overlap
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True

            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                break
            else:
                # Adjust over-exponential growth
                new_radii = radii + (new_radii - radii) * 0.95

        # Apply expanded radii and refine
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())