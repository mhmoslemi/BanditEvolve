import numpy as np

def run_packing():
    n = 26
    cols = int(1 + np.sqrt(n))
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
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Apply geometric reconfiguration through spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply global non-local spatial hashing transformation
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (1.0 + np.random.uniform(-0.3, 0.3))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (1.0 + np.random.uniform(-0.3, 0.3))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
        
        # Apply controlled radius expansion on the circle with minimal constraint tightness
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Calculate constraint tightness based on spatial proximity and edge distances
            dists = np.zeros((n, n))
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            constraint_tightness = np.zeros(n)
            for i in range(n):
                constraint_tightness[i] += (1.0 - v[3*i] - v[3*i+2]) + (1.0 - v[3*i+1] - v[3*i+2])
                for j in range(n):
                    if i != j:
                        constraint_tightness[i] += max(0, radii[i] + radii[j] - dists[i, j])
            
            # Find the circle with the smallest tightness
            least_constrained_idx = np.argmin(constraint_tightness)
            
            # Expand radius while maintaining overlap constraints
            total_sum = np.sum(radii)
            target_total_sum = total_sum + 0.007
            expansion = (target_total_sum - total_sum) / (n - 1)
            
            # Distribute expansion with adaptive perturbation
            new_radii = radii.copy()
            new_radii[least_constrained_idx] += expansion * 1.2  # Slight over-expansion
            for i in range(n):
                if i != least_constrained_idx:
                    new_radii[i] += expansion * (1.0 + 0.1 * np.random.rand())
            
            # Validate and optimize with new radii
            valid_new = True
            while valid_new:
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                
                valid_new = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < new_radii[i] + new_radii[j] - 1e-12:
                            valid_new = False
                            break
                    if not valid_new:
                        break
                
                if not valid_new:
                    # Reduce expansion if constraints fail
                    new_radii = radii + (new_radii - radii) * 0.98
                else:
                    break
            
            v = expanded_v
            
            # Final optimization with updated radii
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())