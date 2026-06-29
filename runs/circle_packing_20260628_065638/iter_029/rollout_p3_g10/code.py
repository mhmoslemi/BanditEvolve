import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    max_iter_first_phase = 1200
    max_iter_second_phase = 600
    radius_tightening_threshold = 0.0005
    boundary_safety_buffer = 1e-8
    perturbation_factor = 0.05
    expansion_weighting = 1.5
    radius_sharing_factor = 0.8
    constraint_tolerance = 1e-10
    spatial_hash_radius_factor = 1.2
    
    def compute_spatial_hash_centers(n, cols, rows):
        """Compute grid-based spatial hash with adaptive offset"""
        xs = []
        ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            x_center = (col + 0.5) / cols
            y_center = (row + 0.5) / rows
            # Add random offset for spatial diversity
            x = x_center + np.random.uniform(-0.08, 0.08)
            y = y_center + np.random.uniform(-0.08, 0.08)
            # Shift alternate rows to create staggered grid
            if row % 2 == 1:
                x += 0.5 / cols
            xs.append(x)
            ys.append(y)
        return xs, ys
    
    def compute_energy_weighted_centers(n, cols, rows, current_radii):
        """Compute centers with weight based on current radii for more dynamic spatial placement"""
        xs = []
        ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            x_center = (col + 0.5) / cols
            y_center = (row + 0.5) / rows
            # Weighted perturbation: smaller circles get more room
            perturbation = np.random.uniform(-0.1 * np.sqrt(current_radii[i]), 0.1 * np.sqrt(current_radii[i]))
            x = x_center + perturbation
            y = y_center + perturbation
            # Shift alternate rows
            if row % 2 == 1:
                x += 0.5 / cols
            xs.append(x)
            ys.append(y)
        return xs, ys
    
    # Initial grid configuration with dynamic perturbations
    cols = 5
    rows = (n + cols - 1) // cols
    xs_initial, ys_initial = compute_spatial_hash_centers(n, cols, rows)
    
    # Initial radii estimate using spatial hashing: smaller circles get more radius
    spatial_radius_scaling = 0.05
    r0 = spatial_radius_scaling / (cols * rows)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs_initial)
    v0[1::3] = np.array(ys_initial)
    v0[2::3] = np.full(n, r0)
    
    # Set up bounds
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # x, y, radius
    
    # Negative objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Create boundary constraints with explicit lambda captures to prevent capture issues
    cons = []
    for i in range(n):
        # X direction
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2] + boundary_safety_buffer})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2] - boundary_safety_buffer})
        # Y direction
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2] + boundary_safety_buffer})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2] - boundary_safety_buffer})
    
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2 + 1e-10
            })
    
    # Initial optimization with focused tolerances and adaptive constraints
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": max_iter_first_phase,
            "ftol": constraint_tolerance,
            "gtol": 1e-8,
            "eps": 1e-8
        }
    )
    
    if res.success:
        # Phase 1: Apply spatial hash and energy weighting with adaptive center calculation
        # First refine based on spatial hash and radius distribution
        v = res.x
        current_radii = v[2::3]
        
        # Apply first-stage spatial hash reconfiguration
        new_xs, new_ys = compute_energy_weighted_centers(n, cols, rows, current_radii)
        perturbed_v = v.copy()
        perturbed_v[0::3] = np.array(new_xs)
        perturbed_v[1::3] = np.array(new_ys)
        
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": max_iter_second_phase,
                "ftol": constraint_tolerance * 0.5,
                "gtol": 1e-8,
                "eps": 1e-8
            }
        )
    
    # Phase 2: Introduce controlled geometric dissection of highly constrained regions
    # First, compute pairwise distances to identify most constrained circle
    if res.success:
        v = res.x
        current_radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distance matrix using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute for each circle, the minimum distance to all others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Circle with least constraint
        most_interacting_idx = np.argmax(min_dists)  # Circle with most interactions
        
        # Calculate current total sum and expansion target
        current_total = current_radii.sum()
        max_possible_growth = 0.008  # Tuned for aggressive expansion without overfitting
        
        # Calculate radius-sharing-based expansion to maintain feasibility
        # Expansion factor for the most constrained circle (least_constrained_idx)
        expansion_factor = 0.9 * max_possible_growth / (n - 1)
        
        # Create expansion vector with targeted expansion
        new_radii = current_radii.copy()
        for i in range(n):
            # Distribute expansion to all
            if i != least_constrained_idx:
                # Use radius-sharing factor to make more aggressive expansion on smaller circles
                # Adjust by inverse radius to emphasize smaller circles (if non-zero)
                new_radii[i] += (expansion_factor * (1.0 + np.log(current_radii[i] + 1e-8) / np.log(1e-4 + 1e-8))) * radius_sharing_factor
        
        # Now attempt to apply expansion while satisfying constraints
        # We'll use a modified vector that respects all constraints
        # Start from a slightly perturbed version to escape local optima
        perturbed_v = v.copy()
        perturbed_v[2::3] = new_radii
        
        # Re-evaluate after expansion, using tighter constraints
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": max_iter_second_phase,
                "ftol": constraint_tolerance * 0.1,
                "gtol": 1e-9,
                "eps": 1e-9
            }
        )
        if res.success:
            v = res.x
            current_radii = v[2::3]
    
    # Final validation and cleanup
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
        
        # Post-processing: add a small spatial hashing to promote diversity
        hash_factor = 0.01 * (np.sqrt(1.0 / radii.sum()))
        spatial_hash = np.random.rand(n, 2) * hash_factor * spatial_hash_radius_factor
        perturbed_v = v.copy()
        perturbed_v[0::3] += spatial_hash[:, 0]
        perturbed_v[1::3] += spatial_hash[:, 1]
        
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 200,
                "ftol": constraint_tolerance * 0.05,
                "gtol": 1e-9,
                "eps": 1e-9
            }
        )
    
    v_out = res.x if res.success else v0
    centers_out = np.column_stack([v_out[0::3], v_out[1::3]])
    radii_out = np.clip(v_out[2::3], 1e-6, None)
    
    return centers_out, radii_out, float(radii_out.sum())