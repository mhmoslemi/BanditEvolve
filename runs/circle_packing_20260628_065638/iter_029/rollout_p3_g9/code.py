import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols  # 6 rows to fit 26 circles in 5 columns

    # Advanced initialization with multi-phase adaptive grid
    # Phase 1: Construct initial grid with staggered rows and spatial jitter
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Primary jitter: small randomized offset
        x = x_center + np.random.uniform(-0.07, 0.07)
        y = y_center + np.random.uniform(-0.07, 0.07)
        
        # Secondary stagger: alternate rows offset to break symmetry
        if row % 2 == 1:
            x += 0.4 / cols  # Larger offset to create spatial dispersion
        
        # Tertiary adjustment for spatial constraint balance
        # Prevent initial clustering in dense rows
        if row < 2:
            x += np.random.uniform(-0.03, 0.03)
            y += np.random.uniform(-0.03, 0.03)

        xs.append(x)
        ys.append(y)
    
    # Phase 2: Adaptive radius assignment based on grid geometry and spatial density
    # Calculate minimum possible radius per circle given grid spacing
    min_possible_radius = 0.4 * (1.0 / (cols + 1))  # Conservative initial radius
    r0 = 0.7 * min_possible_radius  # Start from a fraction of min possible to allow growth
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Constraints: Ensure the bounds list and decision vector length match (3*n)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # (x_low, x_high), (y_low, y_high), (r_low, r_high)

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective: minimize negative of sum to maximize radii

    # Vectorized constraint setup using closure capture with proper i binding
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with tighter control using vectorized logic
    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            # Use vectorized indexing for fast access
            overlap_cons.append({
                "type": "ineq", 
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2
            })
    
    # Combine all constraints
    cons += overlap_cons

    # Initial optimization with increased max iterations and tighter tolerances
    initial_res = minimize(neg_sum_radii, v0, method="SLSQP", 
                          bounds=bounds, constraints=cons, 
                          options={"maxiter": 800, "ftol": 1e-11, "gtol": 1e-11})

    # Phase 3: Adaptive spatial reconfiguration with multi-stage perturbation and feedback
    if initial_res.success:
        # Store initial result for comparison
        v_initial = initial_res.x
        centers_initial = np.column_stack([v_initial[0::3], v_initial[1::3]])
        radii_initial = v_initial[2::3]

        # Generate geometric transformation matrix for reconfiguration
        # Create a matrix of random displacements scaled by radii to create spatial diversity
        spatial_transform = np.random.randn(n, 2) * np.sqrt(0.1 * np.mean(radii_initial))
        perturbed_v = v_initial.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_transform[i, 0] * np.sqrt(radii_initial[i])
            perturbed_v[3*i+1] += spatial_transform[i, 1] * np.sqrt(radii_initial[i])

        # Re-optimized with perturbed positions, increased max iterations
        reconfig_res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", 
                               bounds=bounds, constraints=cons, 
                               options={"maxiter": 1200, "ftol": 1e-11, "gtol": 1e-11})

        # Phase 4: Dynamic constraint re-evaluation and spatial hierarchy adjustment
        if reconfig_res.success:
            v = reconfig_res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            
            # Compute distance matrix with vectorization for performance
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)  # (n,n)
            
            # Identify most constrained (tightest) circle (lowest min distance)
            # This is not just the smallest radius, but the most spatially restricted
            min_dists = np.min(dists, axis=1)
            tightest_idx = np.argmin(min_dists)  # circle with tightest spatial constraints

            # Apply specialized geometric restructuring to the tightest circle
            # Move it to the top-left corner with maximal spatial freedom
            # This helps create new space for other circles to grow
            v[3*tightest_idx] = max(1e-5, v[3*tightest_idx] - 0.1 * radii[tightest_idx])
            v[3*tightest_idx+1] = max(1e-5, v[3*tightest_idx+1] - 0.1 * radii[tightest_idx])
            
            # Re-optimized with adjusted spatial configuration
            adjusted_res = minimize(neg_sum_radii, v, method="SLSQP",
                                   bounds=bounds, constraints=cons, 
                                   options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-11})

            # Phase 5: Targeted Radius Expansion using Global Sum Constraint Feedback
            if adjusted_res.success:
                v = adjusted_res.x
                radii = v[2::3]
                centers = np.column_stack([v[0::3], v[1::3]])
                
                # Compute total current and target radii sum
                current_total = np.sum(radii)
                target_total = current_total + 0.015  # Aim for 1.5% increase in total

                # Identify the least constrained circle to expand
                min_dists = np.min(dists, axis=1)
                least_constrained_idx = np.argmax(min_dists)
                
                # Create expansion vector with focused expansion to least constrained circle
                max_expansion = (target_total - current_total) * 0.8  # Allow 80% to be directed to this circle
                # Avoid overloading with expansion, prevent violating constraints
                safety_buffer = 0.15  # 15% margin to prevent overshooting during optimization
                expansion_factor = max_expansion / (1.0 - safety_buffer)
                
                # Create new_radii vector with expanded least constrained circle
                new_radii = radii.copy()
                new_radii[least_constrained_idx] = np.clip(
                    radii[least_constrained_idx] + expansion_factor * 1.25, 
                    1e-4, 0.45)  # 0.45 is empirically derived upper limit for 26 circles
                
                # Use a more efficient method to compute expansion and apply it
                # This also includes validation to avoid overstepping constraints
                while True:
                    expanded_v = v.copy()
                    expanded_v[2::3] = new_radii
                    expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])

                    # Apply safety check for all circle placements
                    valid = True
                    for i in range(n):
                        for j in range(i+1, n):
                            dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                            dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                            if np.sqrt(dx*dx + dy*dy) < new_radii[i] + new_radii[j] - 1e-8:
                                # Overlap found, reduce expansion
                                new_radii = radii.copy()
                                new_radii[least_constrained_idx] = max(new_radii[least_constrained_idx] - 0.0005, 1e-4)
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        break
                
                # Re-evaluate with expanded radii in optimized configuration
                res = minimize(neg_sum_radii, expanded_v, method="SLSQP",
                               bounds=bounds, constraints=cons,
                               options={"maxiter": 700, "ftol": 1e-11, "gtol": 1e-11})

                if res.success:
                    v = res.x
                    radii = v[2::3]
                
    # Post-optimization validation and clipping (ensure no radius < 1e-6)
    v = res.x if res.success else v_initial
    radii = np.clip(v[2::3], 1e-6, 0.45)  # Max radius is empirically determined for 26 circles
    centers = np.column_stack([v[0::3], v[1::3]])
    return centers, radii, float(radii.sum())