import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols

    # Initialize centers using a staggered grid with added randomization and spatial awareness
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add random offset to break symmetry
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        # Add alternating row shift for staggered pattern
        if row % 2 == 1:
            x += 0.4 / cols
        xs.append(x)
        ys.append(y)
    
    # Determine initial radius based on spacing and safety margin
    base_radius = 0.28 / cols * 0.95
    v0 = np.zeros(3 * n)
    v0[::3] = np.array(xs)  # x positions
    v0[1::3] = np.array(ys)  # y positions
    v0[2::3] = np.full(n, base_radius)  # initial radii

    # Define bounds for all parameters
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Optimization objective function (minimizing negative sum of radii)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint for x-boundary: x - r >= 0
    def x_left_bound(v, i):
        return v[3*i] - v[3*i+2]
    
    # Constraint for x-boundary: x + r <= 1
    def x_right_bound(v, i):
        return 1.0 - v[3*i] - v[3*i+2]
    
    # Constraint for y-boundary: y - r >= 0
    def y_bottom_bound(v, i):
        return v[3*i+1] - v[3*i+2]
    
    # Constraint for y-boundary: y + r <= 1
    def y_top_bound(v, i):
        return 1.0 - v[3*i+1] - v[3*i+2]
    
    # Constraint for circle overlap: dist^2 >= (r1 + r2)^2
    def circle_overlap(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        r1 = v[3*i+2]
        r2 = v[3*j+2]
        return dx*dx + dy*dy - (r1 + r2)**2

    # Build the constraints
    cons = []
    for i in range(n):
        # Add constraints for x-boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: x_left_bound(v, i)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: x_right_bound(v, i)})
        # Add constraints for y-boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: y_bottom_bound(v, i)})
        cons.append({"type": "ineq", "fun": lambda v, i=i: y_top_bound(v, i)})
    
    # Add overlap constraint between all pairs of circles
    for i in range(n):
        for j in range(i+1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: circle_overlap(v, i, j)})

    # First optimization run using SLSQP with increased max iterations
    # Initial optimization with tighter tolerances
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-8, "disp": False}
    )
    
    if res.success:
        # Step 1: Asymmetric reconfiguration
        v = res.x
        
        # Generate a perturbation vector with radius-dependent intensity
        # This creates spatial diversity without violating constraints
        perturbation = np.random.rand(n, 2) * 0.04  # small spatial noise
        perturbation *= np.sqrt(v[2::3])  # intensity scaled by radius
        perturbed_v = v.copy()
        perturbed_v[::3] += perturbation[:, 0]  # x positions
        perturbed_v[1::3] += perturbation[:, 1]  # y positions
        
        # Re-optimize with perturbed parameters to explore new configurations
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-8, "disp": False}
        )

    if res.success:
        # Step 2: Targeted radius expansion
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the least constrained circle by maximizing minimum distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate maximum expansion based on current sum and potential
        current_total = np.sum(radii)
        max_growth = 0.006  # max allowed total radius increase
        expansion_factor = max_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Introduce slight asymmetry in expansion to avoid grid lock
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.1  # slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion = expansion_factor * (1.0 + 0.1 * np.random.rand())  # random variation
                new_radii[i] += expansion
        
        # Apply expansion step-by-step to avoid constraint violations
        for _ in range(5):  # limit steps to avoid infinite loops
            # Test validity
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
            
            if valid:
                break
            else:
                # If invalid, reduce expansion slightly
                new_radii -= 0.1 * expansion_factor
        
        # Apply the optimized radii
        v_expanded = v.copy()
        v_expanded[2::3] = new_radii
        
        # Final optimization to clean up configuration
        res = minimize(
            neg_sum_radii,
            v_expanded,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-8, "disp": False}
        )

    # Final check and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Final validation safety check (optional)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
            if dist < radii[i] + radii[j] - 1e-12:
                raise ValueError(f"Circles {i} and {j} overlap beyond tolerance")
    
    return centers, radii, float(radii.sum())