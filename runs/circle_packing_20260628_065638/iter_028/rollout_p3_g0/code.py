import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with refined stochastic grid layout and adaptive scaling
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Initialize with adaptive spatial scaling to allow for better spread
        scale = 0.8 + 0.2 * np.random.rand()  # Adaptive spatial variance
        dx = np.random.uniform(-0.05 * scale, 0.05 * scale)
        dy = np.random.uniform(-0.05 * scale, 0.05 * scale)
        x = x_center + dx
        y = y_center + dy
        if row % 2 == 1:
            x += 0.5 / cols * scale  # Staggered grid with variable scale
        xs.append(x)
        ys.append(y)
    
    # Compute base radii based on grid and spacing
    base_radius = 0.35 / cols
    r0 = np.random.uniform(0.25 * base_radius, 0.5 * base_radius, size=n) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Define strict bounds that match the length of the optimization vector
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n

    # Define negative sum of radii as objective: minimizing negative sum is equivalent to maximizing sum
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint handling using lambda expressions with captured indices
    cons = []
    for i in range(n):
        # Left side constraint (x - r >= 0)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right side constraint (1.0 - x - r >= 0)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom constraint (y - r >= 0)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top constraint (1.0 - y - r >= 0)
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints using vectorized distance calculations with closed-form
    # Vectorized implementation of distance constraint: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            # Using lambda with captured i and j to avoid lambda closure issues
            # This method uses static capture to ensure correct reference
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with advanced settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "eps": 1e-10})

    # Apply adaptive spatial reconfiguration with gradient-based perturbation
    if res.success:
        v = res.x
        # Spatial hash with gradient-enhanced perturbation for fine-tuning
        grad_perturbation = np.random.rand(n, 2) * 0.03
        grad_factor = np.clip(v[2::3] / np.mean(v[2::3]), 0.5, 1.5)  # Radius-dependent scaling
        perturbed_v = v.copy()
        for i in range(n):
            dx = grad_perturbation[i, 0] * grad_factor[i]
            dy = grad_perturbation[i, 1] * grad_factor[i]
            perturbed_v[3*i] += dx
            perturbed_v[3*i+1] += dy
        
        # Re-evaluate after spatial perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-10})

    # Asymmetric target expansion on least constrained circle with adaptive gradient-based expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute inter-circle distances using vectorized computation
        # Broadcast to compute pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        pairwise_distances = np.sqrt(dx**2 + dy**2)
        
        # Compute closest distances per circle
        min_distances = np.min(pairwise_distances, axis=1)
        # Find circle with maximum distance to other circles
        least_constrained_idx = np.argmax(min_distances)
        
        # Compute total current sum and potential expansion
        current_total = np.sum(radii)
        # Aim for 2% growth based on current radius distribution
        target_growth = 0.002 * current_total
        
        # Create new radius vector with expansion
        new_radii = radii.copy()
        
        # Apply targeted expansion with radius-dependent gradient
        expansion_factor = 0.85 * (target_growth / (n - 1))  # Use 85% of target growth
        expansion_factor += 0.15 * np.random.rand()  # Add stochastic variation
        min_circle_radii = radii[least_constrained_idx]
        max_allowable_growth = (1.0 - min_circle_radii) * 1.2  # Safety margin
        new_radii[least_constrained_idx] = min(min_circle_radii + expansion_factor, 
                                               min_circle_radii + max_allowable_growth)
        
        # Apply controlled expansion across circles for system-wide stability
        for i in range(n):
            if i != least_constrained_idx:
                # Use radius-dependent scaling to increase "weaker" circles more
                expansion = expansion_factor * (radii[i] / radii[least_constrained_idx])
                new_radii[i] += expansion
        
        # Validate the expanded configuration
        # Use brute-force checking with tolerance to avoid numerical errors
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < new_radii[i] + new_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        # If invalid, scale back expansion proportionally
        if not valid:
            # Calculate scaling factor based on minimal violation
            min_violation = np.inf
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    violation = dist - (new_radii[i] + new_radii[j] - 1e-12)
                    if violation < 0:
                        min_violation = max(min_violation, violation)
            expansion_scaling = 0.9 * (np.min(new_radii) / (np.min(new_radii) + 1e-10))
            new_radii = radii + (new_radii - radii) * expansion_scaling
        
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final validation to ensure no overlap with new radii
        if not check_overlap(centers, new_radii):
            # If failed, revert to previous configuration
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
        else:
            v = v_new
            radii = new_radii
            centers = np.column_stack([v[0::3], v[1::3]])
        
        # Run a final refined optimization for stability
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10})

    # Final cleanup and result packaging
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())

def check_overlap(centers, radii):
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True