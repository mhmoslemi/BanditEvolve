import numpy as np

def run_packing():
    n = 26
    cols = 6  # more columns to allow for more flexible grid-like tiling
    rows = np.ceil(n / cols).astype(int)
    
    # Initialize with randomized geometric tiling to break symmetry and allow diverse configurations
    # Create a tiling that randomly samples from a hexagonal or staggered grid pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base center positions based on grid layout
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Random perturbation to break symmetry
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Staggered rows to reduce clustering and allow better packing
        if row % 2 == 1:
            x += 0.4 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radius based on grid spacing with dynamic adjustment
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bound entries for 26 circles

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left boundary constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: 1.0 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: 1.0 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints using broadcasting to ensure all pairs are handled
    # Note: We'll use vectorized computation for pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            # Constraint: distance squared between centers >= (r_i + r_j)^2
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                - (v[3*i+2] + v[3*j+2])**2
            })

    # Run the main optimization with tighter settings
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 1500,
            "ftol": 1e-10,
            "eps": 1e-12,
            "disp": False
        }
    )

    v = res.x if res.success else v0

    # Post-optimization reconfiguration with guided expansion to explore novel configurations
    if res.success:
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3].copy()
        
        # Precompute pairwise distances for constraint checking with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Find the circle with the maximum margin for possible expansion (least constrained in terms of min distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)  # Circle with the largest minimum distance to other circles

        # Calculate the theoretical maximum expansion potential for the least constrained circle
        # This is based on the total current radii and how much more space is available
        curr_total = np.sum(radii)
        max_possible_sum = curr_total + 0.012  # Set a modest target for potential expansion
        expansion_factor = (max_possible_sum - curr_total) / (n - 1)
        
        # Apply an adaptive, geometrically-aware expansion
        # We will increase the least constrained circle and others proportionally
        new_radii = radii.copy()
        # Increase the least constrained circle by a moderate factor
        new_radii[least_constrained_idx] += expansion_factor * 1.15  # slight over-expansion to encourage configuration shift
        # Increase other circles with stochasticity and spatial awareness (closer to larger circles get slightly less)
        for i in range(n):
            if i != least_constrained_idx:
                weight = 1.0 - dists[i, least_constrained_idx] / np.max(dists[i, :]) * 0.3
                new_radii[i] += expansion_factor * (weight + 0.1 * np.random.rand())  # stochastic adjustment

        # Validate the new configuration
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate all pairwise constraints using broadcasting to optimize and avoid scalar loops
            valid = True
            # Compute pairwise distances with broadcasting
            dx_exp = expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0]
            dy_exp = expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1]
            dists_exp = np.sqrt(dx_exp**2 + dy_exp**2)
            for i in range(n):
                for j in range(i + 1, n):
                    if dists_exp[i, j] < (new_radii[i] + new_radii[j]) - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, reduce expansion gradually
                new_radii = radii + (new_radii - radii) * 0.96  # moderate reduction to preserve feasibility

        # Final optimization pass with adaptive constraints and tighter tolerances
        res = minimize(
            neg_sum_radii,
            expanded_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 600,
                "ftol": 1e-11,
                "eps": 1e-12,
                "disp": False
            }
        )

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())