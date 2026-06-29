import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with staggered, randomized grid with increased randomness and refined distribution
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Increased offset range to promote diversity in spacing
        x = x_center + np.random.uniform(-0.1, 0.1)
        y = y_center + np.random.uniform(-0.1, 0.1)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols * 1.2  # Slightly increased stagger to reduce clustering
        # Introduce a small geometric bias to avoid symmetric alignment
        if i % 3 == 0:
            y += np.random.uniform(-0.02, 0.02)
        xs.append(x)
        ys.append(y)
    
    # Start with radius as small fraction of space, adjusted per grid cell
    base_radius = 0.35 / cols
    r0 = base_radius - 1e-3
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for centers and radii
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Construct constraints with explicit closure-binding and vectorization
    cons = []

    # Boundary constraints for all circles
    for i in range(n):
        # Left: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right: 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top: 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints between all pairs
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tight tolerances and large iteration count
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "disp": False})
    
    # Primary optimization step
    if res.success:
        v = res.x
        
        # Randomized geometric hashing for disruptive spatial transformation
        # Weights are designed to influence spatial hashing more in center circles
        hash_weights = np.random.rand(n, 2) * 0.05 + 0.01 * (np.arange(n) < n//2)
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_weights[i, 0] * (1.0 if i % 5 == 0 else 1.0)
            perturbed_v[3*i+1] += hash_weights[i, 1]
        
        # Recalculate constraints based on perturbed configuration (no explicit re-evaluation needed)
        # Use existing constraints with updated perturbed_v
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "disp": False})
    
    # Secondary optimization to refine and enhance
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Find least constrained circle based on minimum distance to others
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Targeted spatial reordering with directional hashing
        # Use directional hashing to create a gradient for expansion
        directional_hash = np.random.rand(n, 2) * 0.08 - 0.04
        directional_weights = np.sin(2 * np.pi * (np.arange(n) / (n)) * 0.6)
        directional_expansion = directional_hash * directional_weights * 0.5
        
        # Prepare for expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += (0.006 / n) * 1.5  # Over-expansion factor
        for i in range(n):
            new_radii[i] += directional_expansion[i, 0] * 0.01  # Controlled adjustment
        
        # Apply expansion with validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
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
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector for final optimization
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization to refine spatial and radius arrangement
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "disp": False})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())