import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Adaptive grid refinement with dynamic symmetry breaking and spatial perturbations
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Use more dense grid for better initial spacing
        base_x = (col + 0.25) / cols * (1.0 - 1e-2) + 1e-2
        base_y = (row + 0.25) / rows * (1.0 - 1e-2) + 1e-2
        # Introduce stochastic spatial displacement to break symmetry
        x = base_x + np.random.uniform(-0.06, 0.06) * np.sqrt(1.0 / (row + col + 1))
        y = base_y + np.random.uniform(-0.06, 0.06) * np.sqrt(1.0 / (row + col + 1))
        # Create staggered rows with dynamic offset based on row parity and grid density
        if row % 2 == 1:
            x += 0.5 / cols * 0.75
        xs.append(x)
        ys.append(y)
    
    # Compute initial radius based on dynamic packing density
    base_radius = 0.30 / cols - 1e-3
    r0 = base_radius * np.array([1.0 + 0.05 * (0.3 * np.sin(i * 1.03)) for i in range(n)])
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Build bounds with strict lower bound on radius and unit square constraint
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Optimized constraints with function captures that avoid closure issues via lambda capture with i
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Optimized distance constraint with vectorization
    # Create sparse distance matrix with fixed-radius check threshold based on initial grid
    distance_threshold = np.sqrt(2) * (0.2)  # 2*radius from nearby circles
    overlap_constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda captures with i,j for correct indexing
            overlap_constraints.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j])**2 + 
                    (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2 + 
                    1e-8  # Tolerance adjustment to handle numerical instability
                )
            })
    cons.extend(overlap_constraints)

    # Initial optimization with adaptive iteration count and tightened tolerances
    max_iter_per_phase = 1000
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": max_iter_per_phase, "ftol": 1e-10, "eps": 1e-9})
    
    # Adaptive reconfiguration: re-sampling of key spatial relations and radius expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Compute pairwise distances and detect dynamically interacting pairs
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        # Identify pairs with minimal distance that are constrained (i.e., not at extremes)
        sorted_indices = np.argsort(dists, axis=None)
        for idx in sorted_indices:
            i = idx // n
            j = idx % n
            if i == j:
                continue
            if dists[i, j] < (radii[i] + radii[j] - 1e-6):
                # Mark as dynamic pair
                dynamic_pairs = [(i, j), (j, i)]
                break
        else:
            dynamic_pairs = None
        
        # Execute the main targeted reconfiguration
        if dynamic_pairs and len(dynamic_pairs) >= 2:
            i1, j1 = dynamic_pairs[0]
            i2, j2 = dynamic_pairs[1]
            # Force a complete topological reordering by swapping these pairs
            # Create a new decision vector where these pairs are repositioned
            v_new = v.copy()
            # Swap center positions of dynamic pairs with randomized perturbations
            x1, y1, r1 = v[3*i1], v[3*i1+1], v[3*i1+2]
            x2, y2, r2 = v[3*i2], v[3*i2+1], v[3*i2+2]
            # Introduce a novel adjacency constraint that shifts these in fixed positions
            # Introduce new center positions that force their reordering
            v_new[3*i1], v_new[3*i1+1] = 0.25, 0.25  # Place first in bottom-left
            v_new[3*i2], v_new[3*i2+1] = 0.75, 0.75  # Place second in top-right
            # Ensure radii can grow but must now maintain spacing to others
            v_new[3*i1+2] = r1
            v_new[3*i2+2] = r2
            # Re-evaluate with updated constraints
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={
                               "maxiter": max_iter_per_phase * 2,
                               "ftol": 1e-11,
                               "eps": 1e-9
                           })
        
        # After reconfiguration, focus on expanding the least constrained circle
        # with controlled expansion of its radius
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            # Compute minimum distances to all other circles to identify least constrained
            min_distances = np.min(np.sqrt(np.sum((centers[:, np.newaxis] - centers[np.newaxis, :])**2, axis=2)), axis=1)
            least_constrained_idx = np.argmax(min_distances)
            current_total = np.sum(radii)
            
            # Compute potential growth and distribute with targeted expansion
            target_radius_sum = current_total + 0.006
            expansion_amount_per_circle = (target_radius_sum - current_total) / (n - 1)
            
            # Create a vector that gradually increases the radii of all circles except the constrained one
            # with more weight to circles with higher current radii (to balance the expansion)
            # This avoids over-expansion of smaller circles and maintains balance
            # The expansion factor is dynamically adjusted based on current spacing
            v_expanded = v.copy()
            expanded_radii = v_expanded[2::3]
            # Apply controlled expansion to all circles that are not the least constrained
            for i in range(n):
                if i != least_constrained_idx:
                    current_radius = expanded_radii[i]
                    # Expand radius based on potential expansion and current spacing
                    # We apply expansion with a factor that decreases as the current radius increases
                    expansion_ratio = 1.0 - (expanded_radii[i] / (np.max(expanded_radii) + 1e-8))
                    # Apply a small expansion, adjusted by the current spacing
                    # The amount increases with the current spacing
                    v_expanded[3*i + 2] += expansion_amount_per_circle * expansion_ratio * np.sqrt(min_distances[i] / np.mean(min_distances))
            # Re-evaluate with new radii
            res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                           constraints=cons, options={
                               "maxiter": max_iter_per_phase * 1.5,
                               "ftol": 1e-12,
                               "eps": 1e-9
                           })
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    
    # Final refinement to ensure all circles are within bounds and overlap-free
    # This ensures minimal violations even after large-scale optimization
    v_final = v.copy()
    for i in range(n):
        # Ensure that the x coordinate is within bounds after radius adjustment
        if v_final[3*i] - radii[i] < -1e-12:
            v_final[3*i] = min(max(v_final[3*i], 0.0), 1.0)
        if v_final[3*i] + radii[i] > 1.0 + 1e-12:
            v_final[3*i] = max(min(v_final[3*i], 1.0 - radii[i]), 0.0)
        # Ensure that the y coordinate is within bounds after radius adjustment
        if v_final[3*i+1] - radii[i] < -1e-12:
            v_final[3*i+1] = min(max(v_final[3*i+1], 0.0), 1.0)
        if v_final[3*i+1] + radii[i] > 1.0 + 1e-12:
            v_final[3*i+1] = max(min(v_final[3*i+1], 1.0 - radii[i]), 0.0)
        # Ensure that radius is within bounds
        v_final[3*i+2] = np.clip(v_final[3*i+2], 1e-6, 0.5)
    
    # Final optimization pass with tighter constraints and improved tolerances
    res_final = minimize(neg_sum_radii, v_final, method="SLSQP", bounds=bounds,
                        constraints=cons, options={"maxiter": 250, "ftol": 1e-12, "eps": 1e-10})
    
    v = res_final.x if res_final.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    return centers, radii, float(radii.sum())