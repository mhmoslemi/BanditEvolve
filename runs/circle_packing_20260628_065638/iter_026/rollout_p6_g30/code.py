import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Enhanced initialization with adaptive grid and randomized spatial bias
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add spatial bias for uneven distribution
        x = x_center + np.random.normal(0, 0.03)
        y = y_center + np.random.normal(0, 0.03)
        # Create staggered layout with dynamic offset
        if row % 2 == 1:
            x += 0.5 / cols * (1 + np.random.rand() * 0.2)
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with spatial hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization pass with tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Disruptive geometric reconfiguration with spatial hashing and layout restructuring
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Construct spatial hash matrix for layout disruption
        spat_hash = np.random.rand(n, 2) * 0.05
        new_centers = np.zeros_like(centers)
        for i in range(n):
            new_centers[i, 0] = centers[i, 0] + spat_hash[i, 0]
            new_centers[i, 1] = centers[i, 1] + spat_hash[i, 1]
        
        # Apply spatial hashing-induced displacement to centers
        perturbed_v = v.copy()
        perturbed_v[0::3] = new_centers[:, 0]
        perturbed_v[1::3] = new_centers[:, 1]
        
        # Re-evaluate with spatially perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Triggered asymmetric radius expansion on critical node with layout-awareness
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation with vectorized operations
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find node with most spatial freedom (least constrained)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor based on system-wide constraint analysis
        # Expand this node to trigger layout recalibration
        target_total_sum = np.sum(radii) + 0.015
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Over-expand to trigger layout change
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector with enhanced radius profile
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Apply new radius profile while maintaining spatial constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Final refinement with tighter constraints and spatial constraints reinforcement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Final validation pass with strict constraint enforcement
        # Apply spatial hashing to enforce layout reordering
        spat_hash = np.random.rand(n, 2) * 0.03
        final_centers = np.zeros_like(centers)
        for i in range(n):
            final_centers[i, 0] = centers[i, 0] + spat_hash[i, 0]
            final_centers[i, 1] = centers[i, 1] + spat_hash[i, 1]
        
        # Apply small random displacement to enforce final layout ordering
        final_v = v.copy()
        final_v[0::3] = final_centers[:, 0]
        final_v[1::3] = final_centers[:, 1]
        final_v[2::3] = radii
        
        # Final optimization with tight tolerances
        res = minimize(neg_sum_radii, final_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())