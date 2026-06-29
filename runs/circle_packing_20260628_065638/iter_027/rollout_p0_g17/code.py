import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with staggered grid + perturbed positions
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce controlled jitter and row staggering for better space utilization
        x = x_center + np.random.uniform(-0.09, 0.09)
        y = y_center + np.random.uniform(-0.06, 0.06)
        if row % 2 == 1:
            x += 0.5 / cols * 1.1
        xs.append(x)
        ys.append(y)
    
    r0 = 0.4 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure bounds list has length 3*n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Use named lambda closures to avoid closure capture issues
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized distance constraint function with lambda closures
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # First-phase optimization with enhanced precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-10})

    # Advanced spatial reconfiguration: geometric hashing with adaptive scaling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate a random geometric hash to perturb the layout
        hash_factor = 0.03
        spatial_hash = np.random.rand(n, 2) * hash_factor
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Second-phase optimization with refined constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-10})

    # Topological reordering and enhanced radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Efficient distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the circle with the smallest and largest radii for targeted operations
        min_radius_idx = np.argmin(radii)
        max_radius_idx = np.argmax(radii)
        min_radius = radii[min_radius_idx]
        max_radius = radii[max_radius_idx]
        
        # Calculate expansion based on current sum and potential
        current_sum = np.sum(radii)
        target_growth = 0.008
        expansion_factor = (target_growth / current_sum) * np.sum(radii) if current_sum != 0 else 0
        
        # Apply targeted expansion on the smallest radius circle
        new_radii = radii.copy()
        new_radii[min_radius_idx] += expansion_factor * 1.3  # Slight over-expansion
        for i in range(n):
            if i != min_radius_idx:
                # Add stochastic expansion with adaptive scaling
                new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())
        
        # Validate and refine expanded radii
        valid = False
        iterations = 0
        max_iterations = 3
        while not valid and iterations < max_iterations:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate configuration
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
                # If overlap, dampen expansion
                new_radii = radii + (new_radii - radii) * 0.96
                iterations += 1
        
        # Update decision vector with validated configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization stage
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())