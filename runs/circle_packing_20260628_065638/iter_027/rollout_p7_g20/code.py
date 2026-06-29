import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    max_iter = 1500
    ftol = 1e-10
    final_ftol = 1e-12
    expansion_factor = 0.5
    radius_shake = 0.03
    position_shake = 0.05
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.45 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i
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
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": max_iter, "ftol": ftol})
    
    # First shake: Perturb the smallest circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Identify the smallest circles for shake
        indices = np.argsort(radii)  # sorted from smallest to largest
        smallest_indices = indices[:4]  # shake the 4 smallest circles
        v_shake = v.copy()

        for idx in smallest_indices:
            # Shake position: add small delta
            x_shake = np.random.uniform(-position_shake, position_shake)
            y_shake = np.random.uniform(-position_shake, position_shake)
            v_shake[3*idx] += x_shake
            v_shake[3*idx+1] += y_shake
            # Shake radius: add small delta
            r_shake = np.random.uniform(-radius_shake, radius_shake)
            v_shake[3*idx+2] += r_shake
        
        # Re-optimizing with new configuration
        res = minimize(neg_sum_radii, v_shake, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": ftol})
    
    # Final targeted radius expansion with control and constraint validation
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute distances using vectorized approach
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate for each circle the minimal distance to other circles
        min_dist_to_other = np.min(dists, axis=1)
        
        # Determine the circles with the largest "margin" (i.e., less constrained)
        # Sort by margin, expand on the top 5 (most unconstrained) circles
        indices = np.argsort(min_dist_to_other)  # largest margin first
        candidates = indices[:8]  # target expansion on most unconstrained circles

        # Compute the current total sum
        current_total = np.sum(radii)
        # Targeting an expansion of 0.007 relative to current
        target_total = current_total + max(0.007, 0.001)

        # Calculate the possible expansion for each of the target circles
        expansion_per_circle = (target_total - current_total) / (n)
        
        # Allocate expansions with some randomness
        expansion_weights = np.random.rand(n)
        expansion_weights[candidates] *= 1.3
        expansion_weights /= np.sum(expansion_weights)
        
        # Calculate how much we can expand without violating constraints
        expansion = np.zeros(n)
        for i in candidates:
            # Only expand if it's not already at max
            max_expansion = expansion_per_circle * expansion_weights[i]
            # Estimate how much we can expand while maintaining constraints
            # Use a safe, gradient-based expansion
            expansion[i] = max_expansion
            # Reduce expansion if nearby circles are small
            for j in range(n):
                if j == i:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                # If too close, limit expansion
                if dist < radii[i] + 0.01:
                    expansion[i] *= 0.5
        
        # Apply expansion with constraint validation
        v_expanded = v.copy()
        v_expanded[2::3] += expansion
        centers_expanded = np.column_stack([v_expanded[0::3], v_expanded[1::3]])
        radii_expanded = v_expanded[2::3]

        # Validate expanded configuration
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers_expanded[i, 0] - centers_expanded[j, 0]
                dy = centers_expanded[i, 1] - centers_expanded[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii_expanded[i] + radii_expanded[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            v = v_expanded
        else:
            # If invalid, reduce expansion slightly
            v = v.copy()
            v[2::3] += expansion * 0.8
        
        # Refine the final configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": final_ftol})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())