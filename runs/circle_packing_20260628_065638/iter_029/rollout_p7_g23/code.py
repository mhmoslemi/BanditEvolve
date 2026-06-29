import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with refined staggered grid and dynamic spacing
    # Adaptive spacing that prevents clustering and enables optimal expansion
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        col_center = (col + 0.5) / cols
        row_center = (row + 0.5) / rows
        # Base positional adjustment based on row/col spacing and random offset
        x = col_center + np.random.uniform(-0.01, 0.01)
        y = row_center + np.random.uniform(-0.01, 0.01)
        # Row-dependent staggered offset to avoid direct overlap
        if row % 2 == 1:
            x += 0.45 / cols  # Staggered offset to break row alignment
            if col % 2 == 1:
                y += 0.03
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation with tighter bounds and dynamic scaling
    base_radius = 0.425 / cols
    r0 = base_radius * (1.0 - (n / 100))  # Radius scales with fewer circles
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.48)]  # Slight upper bound for radius to control growth

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create vectorized bounds and constraints with closure handling
    def make_bound_func(i, axis):
        if axis == 0:
            return lambda v: v[3*i] - v[3*i + 2]  # Left x + radius >= 0
        elif axis == 1:
            return lambda v: 1.0 - v[3*i] - v[3*i + 2]  # Right x - radius <= 1
        elif axis == 2:
            return lambda v: v[3*i + 1] - v[3*i + 2]  # Bottom y + radius >= 0
        elif axis == 3:
            return lambda v: 1.0 - v[3*i + 1] - v[3*i + 2]  # Top y - radius <= 1
        else:
            raise ValueError("Invalid axis for bound function")

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: make_bound_func(i, 0)(v)})  # Left
        cons.append({"type": "ineq", "fun": lambda v, i=i: make_bound_func(i, 1)(v)})  # Right
        cons.append({"type": "ineq", "fun": lambda v, i=i: make_bound_func(i, 2)(v)})  # Bottom
        cons.append({"type": "ineq", "fun": lambda v, i=i: make_bound_func(i, 3)(v)})  # Top
        
    # Vectorized overlap constraints with precomputed pairs and explicit gradient
    # We compute pairwise distances and use vectorized operations for speed
    from itertools import combinations
    overlap_pairs = list(combinations(range(n), 2))
    
    def compute_pair_overlap(v):
        # Extract centers and radii
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        
        # Create distance matrix for all pairs
        dist_sq = np.zeros((n, n))
        for i, j in overlap_pairs:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq[i, j] = dx * dx + dy * dy
            dist_sq[j, i] = dist_sq[i, j]
        
        # Overlap condition: dist^2 < (r_i + r_j)^2
        overlap_mask = np.triu((dist_sq < (rs + rs[:, np.newaxis])**2)).astype(int)
        # Sum the violations: if any overlapping pair is violating, return negative
        violation_count = np.sum(overlap_mask)
        return -violation_count
    
    def compute_pair_overlap_jac(v):
        # Jacobian: derivative of the constraint function
        n = len(v) // 3
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        
        jac = np.zeros_like(v)
        for k in range(n):
            # Derivative with respect to x_k
            for (i, j) in overlap_pairs:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = dx * dx + dy * dy
                # Gradient w.r.t x_k
                if i == k:
                    # d/dx_k (d - (r_i + r_j)^2)
                    jac[3*k] -= 2 * (dx - (rs[i] + rs[j]) * (2 * rs[i] * (1 if i == k else 0) + 2 * rs[j] * (1 if j == k else 0))) / (2 * np.sqrt(d))
                elif j == k:
                    jac[3*k] -= 2 * (dx) * (2 * rs[j] * (1 if j == k else 0) + 2 * rs[i] * (1 if i == k else 0)) / (2 * np.sqrt(d))
                # Similarly for y_k
                if i == k:
                    jac[3*k+1] -= 2 * (dy) * (2 * rs[i] * (1 if i == k else 0) + 2 * rs[j] * (1 if j == k else 0)) / (2 * np.sqrt(d))
                elif j == k:
                    jac[3*k+1] -= 2 * (dy) * (2 * rs[j] * (1 if j == k else 0) + 2 * rs[i] * (1 if i == k else 0)) / (2 * np.sqrt(d))
                # Radius derivatives
                if i == k:
                    jac[3*k+2] -= 2 * (2 * rs[i] * (1 if i == k else 0) + 2 * rs[j] * (1 if j == k else 0)) * dx / (2 * np.sqrt(d)) * (1)
                if j == k:
                    jac[3*k+2] -= 2 * (2 * rs[j] * (1 if j == k else 0) + 2 * rs[i] * (1 if i == k else 0)) * dx / (2 * np.sqrt(d)) * (1)
        # We're minimizing -overlap violations, so Jacobian signs are preserved
        return jac

    for i, j in overlap_pairs:
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i, j=j: compute_pair_overlap(v),
                     "jac": lambda v, i=i, j=j: compute_pair_overlap_jac(v)})

    # Initial optimization
    res = minimize(
        neg_sum_radii, 
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 700,
            "ftol": 1e-12,
            "gtol": 1e-10,
            "eps": 1e-10,
            "disp": False,
            "iprint": -1,
            "maxcor": 100
        }
    )

    # Post-optimization refinement
    if res.success:
        v = res.x

        # Adaptive perturbation: find and isolate top 2 most dynamically interacting circles
        # Use pairwise distance-based interaction scores for this
        xs_ = v[0::3]
        ys_ = v[1::3]
        rs_ = v[2::3]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    dx = xs_[i] - xs_[j]
                    dy = ys_[i] - ys_[j]
                    dists[i, j] = np.sqrt(dx**2 + dy**2)
                    dists[j, i] = dists[i, j]
        # Compute interaction score: sum of inverse distance to others
        interaction_scores = np.sum(1 / (dists + 1e-12), axis=1)
        top_two_indices = np.argsort(interaction_scores)[-2:]
        
        # Reconfigure the top two most active circles by applying explicit geometric dissection
        top_index, second_index = top_two_indices
        # Create new positions for top two circles with enforced spacing > 0.05 to force topological reordering
        # These circles will be at specific, well-defined locations to avoid overlap
        # Enforce a fixed offset between them to create a dissection
        new_top_x = np.random.uniform(0.2, 0.8)
        new_top_y = np.random.uniform(0.2, 0.6)
        new_second_x = new_top_x + 0.08
        new_second_y = new_top_y

        # Apply this new configuration to the system, keeping other circles fixed
        new_v = v.copy()
        new_v[3 * top_index] = new_top_x
        new_v[3 * top_index + 1] = new_top_y
        new_v[3 * second_index] = new_second_x
        new_v[3 * second_index + 1] = new_second_y

        # Re-run on this new configuration
        res = minimize(
            neg_sum_radii,
            new_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 300,
                "ftol": 1e-12,
                "gtol": 1e-10,
                "eps": 1e-10,
                "disp": False,
                "iprint": -1,
                "maxcor": 100
            }
        )

    # After reconfiguration, identify the "least constrained" circle (most separated from others)
    # This is not just the one with smallest radius, but the one that's spatially most separate
    # Again, we find this via distance metrics
    if res.success:
        v = res.x
        # Extract coordinates and radii
        xs_ = v[0::3]
        ys_ = v[1::3]
        rs_ = v[2::3]
        # Compute all pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    dx = xs_[i] - xs_[j]
                    dy = ys_[i] - ys_[j]
                    dists[i, j] = np.sqrt(dx**2 + dy**2)
                    dists[j, i] = dists[i, j]
        # Compute spatial constraint metric: inverse of minimum distance
        spatial_constraints = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(spatial_constraints)  # Circle that is most isolated

        # Now, we apply a controlled expansion to this circle while preserving constraints
        # Use a dynamic expansion factor that considers the current configuration and potential
        # Also introduce a new, artificial adjacency constraint between this circle and the least dense area
        current_r = rs_[least_constrained_idx]
        # Find the region of empty space to enforce adjacency
        # For this, we define a target area that hasn't been occupied by large circles
        target_area_radius = 0.02
        target_area_center = np.array([0.5, 0.5])  # Center square, avoid overlaps with main cluster
        # Compute distance from this circle to the target area
        dx = xs_[least_constrained_idx] - target_area_center[0]
        dy = ys_[least_constrained_idx] - target_area_center[1]
        dist_to_target_area = np.sqrt(dx**2 + dy**2)
        # Distance to target area is not the only constraint, but helps form the new interaction
        # We enforce a minimum distance to the area of interest, using a soft constraint
        # We'll add a soft constraint that maintains this distance >= (current radius + target_radius)
        # This introduces a new dynamic adjacency to the system
        # But for the purpose of expansion, we can use this as a way to control the expansion of the least constrained circle
        
        # Estimate a maximum possible expansion for this circle
        # This is derived from the minimum distance to all other circles and existing geometry
        min_dist_to_others = np.min(dists[least_constrained_idx, :])
        potential_max_growth = (min_dist_to_others - rs_[least_constrained_idx]) * 0.8
        # Compute a growth multiplier (can be 1.0 for direct expansion)
        growth_multiplier = 1.05  
        # Compute how much we can expand to meet target_total_sum (from before)
        # We first compute the current total sum to establish target growth
        current_total_sum = np.sum(rs_)
        target_total_sum = current_total_sum + 0.009  # 0.009 additional sum is a goal
        total_grow = target_total_sum - current_total_sum
        # To distribute this increase, we expand the least constrained by a larger amount
        # While keeping the constraint from being violated by over-expansion
        max_expansion = np.min([potential_max_growth, total_grow * 1.2])
        # Apply expansion to the least constrained circle
        expansion_amount = max_expansion * growth_multiplier
        # To make this even more strategic, we apply it as a direct expansion and re-run
        new_v = v.copy()
        new_v[3*least_constrained_idx + 2] = current_r + expansion_amount
        
        # Re-optimize with this new configuration
        res = minimize(
            neg_sum_radii,
            new_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 200,
                "ftol": 1e-12,
                "gtol": 1e-10,
                "eps": 1e-10,
                "disp": False,
                "iprint": -1,
                "maxcor": 100
            }
        )

        # After optimization, perform final refinement by checking for edge cases and rechecking bounds
        if res.success:
            v = res.x
            # Enforce boundary checks (this is critical because the solver may not do it perfectly)
            for i in range(n):
                x, y, r = v[3*i], v[3*i+1], v[3*i+2]
                # Check if x - r is less than -1e-12
                if x - r < -1e-12:
                    v[3*i] = max(min(x, 1.0), 0.0)
                # Check if x + r > 1.0 + 1e-12
                if x + r > 1.0 + 1e-12:
                    v[3*i] = max(min(x, 1.0 - r), 0.0)
                # Similarly for y
                if y - r < -1e-12:
                    v[3*i+1] = max(min(y, 1.0), 0.0)
                if y + r > 1.0 + 1e-12:
                    v[3*i+1] = max(min(y, 1.0 - r), 0.0)
                # Check if radius is within bounds (we already have bounds, but do final clip)
                v[3*i+2] = np.clip(v[3*i+2], 1e-4, 0.48)

            # Final optimization with very tight tolerances to settle any remaining numerical issues
            res = minimize(
                neg_sum_radii,
                v,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={
                    "maxiter": 150,
                    "ftol": 1e-12,
                    "gtol": 1e-10,
                    "eps": 1e-10,
                    "disp": False,
                    "iprint": -1,
                    "maxcor": 100
                }
            )

    # Final result preparation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-4, 0.48)
    return centers, radii, float(radii.sum())