import numpy as np

def run_packing():
    n = 26
    cols = 6  # Adjust to 6 based on geometric optimization for better spacing
    rows = (n + cols - 1) // cols
    
    # Adaptive spatial initialization with clustered geometric hashing and symmetry breaking
    xs = []
    ys = []
    spatial_hash = np.random.rand(n, 3)  # Add third dimension for hashing
    hash_factor = 0.02  # Adjusted to better avoid symmetry
    for i in range(n):
        # Row and col: adapt rows with sqrt(n) to ensure even spacing
        row = i // cols
        col = i % cols
        # Calculate base positions with geometric scaling
        base_x = (col + 0.25) / cols
        base_y = (row + 0.25) / rows
        # Add adaptive randomized offset to break symmetry
        rand_x = 0.08 * np.sin(4 * np.pi * (base_x + spatial_hash[i, 0]))
        rand_y = 0.08 * np.cos(4 * np.pi * (base_y + spatial_hash[i, 1]))
        x = base_x + rand_x
        y = base_y + rand_y
        # Alternate row shifting with adaptive spacing
        if row % 2 == 1:
            shift_x = 0.25 / cols * (1 - np.sin(10.0 * spatial_hash[i, 2]))
            x += shift_x
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.28 / cols - 2e-3  # Reduced base radius for more precise exploration
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Strict bounds with enhanced safety, ensuring 3*n length for all arrays
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)]  # Reduced radius lower bound to 1e-5

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Optimization objective remains same

    # Vectorized constraints with lambda capturing i, ensuring consistent handling across iterations
    cons = []
    for i in range(n):
        # Left bound: x - r => 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right bound: 1 - x - r => 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom bound: y - r => 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top bound: 1 - y - r => 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using lambda closures to prevent capture issues, now with tighter tolerances
    for i in range(n):
        for j in range(i + 1, n):
            # Constraint: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
            cons.append({
                "type": "ineq", 
                "fun": (lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                        - (v[3*i+2] + v[3*j+2])**2
                )
            })

    # First-phase optimization with enhanced convergence parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-12, "gtol": 1e-10})

    # If optimization failed, ensure bounds and constraints are rechecked
    if not res.success:
        # Reinitialize with modified base position strategy for failed convergence
        xs = []
        ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            base_x = (col + 0.2) / cols
            base_y = (row + 0.2) / rows
            # Add adaptive random offset based on spatial hashing
            rand_offset = np.random.uniform(-0.04, 0.04, size=2)
            x = base_x + rand_offset[0]
            y = base_y + rand_offset[1]
            if row % 2 == 1:
                x += 0.2 / cols * (1 - np.cos(20.0 * i))
            xs.append(x)
            ys.append(y)
        v0 = np.empty(3 * n)
        v0[0::3] = np.array(xs)
        v0[1::3] = np.array(ys)
        v0[2::3] = np.full(n, r0)
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1200, "ftol": 1e-12, "gtol": 1e-10})

    # Structural perturbation for non-local exploration (geometric hashing-based)
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Apply spatial hash-based geometric perturbation with adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.04  # Smaller perturbation for stability
        # Use radii-based scaling for spatial hashing
        # For non-adjacent circles only: apply spatial perturbation to maximize configuration diversity
        non_adjacent_indices = np.random.choice(n, size=n // 2, replace=False)
        for idx in non_adjacent_indices:
            x_idx = 3*idx
            y_idx = 3*idx + 1
            x_perturb = spatial_hash[idx, 0] * (radii[idx] / np.mean(radii))
            y_perturb = spatial_hash[idx, 1] * (radii[idx] / np.mean(radii))
            v[x_idx] += x_perturb
            v[y_idx] += y_perturb
            # Apply bounds after perturbation
            if v[x_idx] < 1e-6:
                v[x_idx] = 1e-6
            if v[x_idx] > 1.0 - 1e-6:
                v[x_idx] = 1.0 - 1e-6
            if v[y_idx] < 1e-6:
                v[y_idx] = 1e-6
            if v[y_idx] > 1.0 - 1e-6:
                v[y_idx] = 1.0 - 1e-6
        # Re-optimization to refine new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-12})

    # Targeted edge-aware radius expansion: optimize for circles near edges
    if res.success:
        v = res.x
        # Calculate edge distances for all circles
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        edge_distances = np.zeros(n)
        for i in range(n):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            edge_distances[i] = min(
                x - r,  # Left
                1.0 - x - r,  # Right
                y - r,  # Bottom
                1.0 - y - r  # Top
            )
        # Identify circles with maximum edge distance (least constrained)
        max_edge_idx = np.argmax(edge_distances)
        # Get current total radius sum
        current_sum = radii.sum()
        # Compute a target expansion to achieve a specific growth
        target_growth = 0.006  # Adjusted to 0.006 for finer control
        expansion_ratio = target_growth / current_sum  # Fractional expansion
        # Ensure expansion doesn't cause overlap with adjacent circles
        expansion_candidates = []
        for i in range(n):
            if i != max_edge_idx:
                # Compute minimal required radius to allow expansion
                # First compute new radius based on expansion ratio
                new_radius = radii[i] + expansion_ratio * (radii[i] - 1e-5)
                if new_radius <= 0.5:  # Avoid hitting maximum safe radius
                    expansion_candidates.append(i)
        # If we have valid candidates, we can consider targeted expansion
        if len(expansion_candidates) > 0:
            # Compute a soft expansion factor based on relative edge distance
            expansion_factors = np.zeros(n)
            expansion_factors[max_edge_idx] = 1.4  # Over-expand the edge circle more
            for i in expansion_candidates:
                expansion_factors[i] = 1.2  # Moderate growth for others
            # Apply expansion with constraint validation
            while True:
                expanded_v = v.copy()
                expanded_v[2::3] = radii + expansion_factors * (radii - 1e-5)
                # Validate configuration
                valid = True
                # Check all pairs
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = expanded_v[3*i] - expanded_v[3*j]
                        dy = expanded_v[3*i+1] - expanded_v[3*j+1]
                        dist = dx*dx + dy*dy
                        if dist < (expanded_v[3*i+2] + expanded_v[3*j+2])**2 - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
                else:
                    # If not valid, scale down the expansion
                    expansion_factors *= 0.95
        # Finally apply any valid expansion and re-verify
        res = minimize(neg_sum_radii, expanded_v if 'expanded_v' in locals() else v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-12})

    # Final validation and output
    v = res.x if res.success else v0
    # Final clipping to ensure valid radii
    radii = np.clip(v[2::3], 1e-6, 0.5)  # Tightened max to 0.5 for stability
    centers = np.column_stack([v[0::3], v[1::3]])
    sum_radii = float(radii.sum())
    return centers, radii, sum_radii