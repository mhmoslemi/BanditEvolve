import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with geometrically optimized base grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add asymmetric offset for diversity in initial positions
        x_offset = np.random.uniform(-0.06, 0.06)
        y_offset = np.random.uniform(-0.06, 0.06)
        x = x_center + x_offset * 2
        y = y_center + y_offset * 2
        # Staggered grid for reduced density in alternate rows
        if row % 2 == 1:
            x += 0.4 / cols
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
        # Left boundary (x - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary (x + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary (y - r >= 0)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary (y + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization stage with aggressive tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-8})

    # Asymmetric reconfiguration with randomized spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Asymmetric spatial reconfiguration using stochastic hashing
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Apply stochastic perturbation with direction and magnitude
            perturbed_v[3*i] += spatial_hash[i, 0] * 0.05
            perturbed_v[3*i+1] += spatial_hash[i, 1] * 0.05
        
        # Re-evaluate using same constraints but with new geometry
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-8})

    # Targeted expansion of least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute inter-circle distances efficiently
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle with minimum minimum distance
        min_dist_per_circle = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dist_per_circle)
        
        # Calculate current total sum
        current_total = np.sum(radii)
        # Set target total sum with cautious expansion
        target_total = current_total + 0.005
        
        # Expand least constrained circle with soft constraints
        new_radii = radii.copy()
        # Give it 50% more expansion potential than others
        expansion_factor = (target_total - current_total) / (n - 0.5)
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        
        # Re-validate with adjusted radii through constraint propagation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True
            
            for i in range(n):
                for j in range(i+1, n):
                    dx_ = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_ = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_**2 + dy_**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, apply softer expansion
                new_radii = radii + (new_radii - radii) * 0.98

        # Update decision variables and re-run with updated radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-8})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())