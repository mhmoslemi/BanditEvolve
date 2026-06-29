import numpy as np

def run_packing():
    n = 26
    cols = 6  # Slightly more columns to allow for more flexible layout
    
    # Generate seed positions using hexagonal tiling with randomized perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / ((n + cols - 1) // cols)
        # Add small randomized perturbation for spatial diversity
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Apply staggered row shift for better packing density
        if row % 2 == 1:
            y += 0.25 / ((n + cols - 1) // cols)
        xs.append(x)
        ys.append(y)
    
    # Set initial radii based on spacing with tighter bounds
    r0 = 0.48 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)  # x positions
    v0[1::3] = np.array(ys)  # y positions
    v0[2::3] = np.full(n, r0)  # radii

    bounds = []
    for _ in range(n):
        # Ensure all constraints have 3*n elements
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # x, y, radius

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective is to minimize the negative of sum of radii

    # Vectorized boundary constraints with lambda closure capture
    cons = []
    for i in range(n):
        # Left edge: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right edge: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom edge: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top edge: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with lambda closure capture
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                              (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                              - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization pass
    res = minimize(neg_sum_radii, v0, method="SLSQP", 
                   bounds=bounds, constraints=cons, 
                   options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-12})

    # Spatial reconfiguration with geometric hashing and dynamic radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash with adaptive perturbation based on spatial distribution
        spatial_hash = np.random.rand(n, 2) * 0.05 / (np.std(centers, axis=0) + 1e-6)
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 800, "ftol": 1e-11, "eps": 1e-13})

    # Targeted expansion of circle with minimal non-zero radius using dynamic growth constraint
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify circle with the least minimal distance to others (most underutilized)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Minimizing min distance indicates underutilized
        
        # Calculate expansion with dynamic total-sum constraint
        current_total = np.sum(radii)
        # Define an adaptive growth target based on spatial distribution and current density
        target_total_sum = current_total + 0.01 * (1 + np.std(radii) / np.mean(radii))
        
        # Ensure expansion is feasible and maintain spatial constraints
        expansion_factor = (target_total_sum - current_total) / (n - 1) * (1 + 0.1 * np.random.rand())  # add variability
        
        # Apply expansion with spatial-aware adjustments
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.1  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.05 * np.random.rand())  # Slight stochastic expansion
        
        # Validate and apply expansion
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate the expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-8:  # Tighter tolerance for stability
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion in a controlled manner
                new_radii = radii + (new_radii - radii) * 0.95  # Moderate reduction

        # Final optimization with tighter tolerances
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-13})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())