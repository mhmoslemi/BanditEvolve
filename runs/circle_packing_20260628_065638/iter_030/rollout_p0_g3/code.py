import numpy as np

def run_packing():
    n = 26
    # Use a more efficient and refined grid with asymmetric column count for better utilization
    def compute_optimized_grid(n):
        # Heuristic for minimal column count to allow asymmetric spatial density
        max_col = int(np.ceil(np.sqrt(n)))
        # Test both min and max column to see which allows better circle packing
        # Try min_col based on density and max_col to maximize possible area coverage
        min_col = max(2, int(np.floor(np.sqrt(n))))
        for candidate_col in [min_col, max_col, min_col + 1, max_col - 1]:
            if candidate_col * 6 < n:
                continue  # avoid too sparse
            rows = (n + candidate_col - 1) // candidate_col
            yield (candidate_col, rows)
    # Select the most optimal candidate grid that maximizes spacing potential
    optimized_cols_rows = list(compute_optimized_grid(n))
    optimized_cols_rows.sort(key=lambda cr: (cr[0] * cr[1]) * np.sqrt(n))  # higher area product
    best_cols, best_rows = optimized_cols_rows[0]
    
    # Initialize positions with structured base and asymmetric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // best_cols
        col = i % best_cols
        base_x = (col + 0.5)/best_cols
        base_y = (row + 0.5)/best_rows
        # Apply asymmetric perturbation for grid refinement
        # Increase perturbation for higher rows to allow vertical expansion
        # Introduce a row-wise scaling to handle uneven row height due to column count
        row_ratio = (row + 0.5) / best_rows  # normalize row index
        row_squeeze = np.clip(row_ratio + 0.05, 0.1, 1.0)
        # Add asymmetric perturbations with higher variation for upper rows
        x_off = np.random.uniform(-0.06 * row_ratio, 0.06 * row_ratio)
        y_off = np.random.uniform(-0.06 * row_ratio, 0.06 * row_ratio) * 2
        # Staggered rows with adaptive offset
        if row % 2 == 1:
            # Create a shifted stagger for odd rows to prevent square patterns
            base_x += 0.5 / best_cols * 0.8
        x = base_x + x_off
        y = base_y + y_off
        # Ensure all positions are bounded within the square
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / best_cols * np.sqrt(n) / n - 1e-3  # Adjust initial radius for grid geometry
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n, matches v
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraint setup with fixed-index access
    cons = []
    for i in range(n):
        # Left: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right: 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top: 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Optimized overlap constraints with vectorized handling and early termination
    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            # Constraint: distance between centers >= radii[i] + radii[j]
            overlap_cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2
            })
    cons.extend(overlap_cons)
    
    # Initial optimization with tighter tolerance and more iterations
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 600, 
            "ftol": 1e-11,  # Tight tolerance for precision
            "gtol": 1e-11,
            "eps": 1e-9,
            "disp": False,
            "iprint": -1
        }
    )
    
    # Safety: Ensure convergence
    if not res.success:
        # Re-run optimization with improved initial guess based on grid
        # Perturb with a geometric grid hash that preserves column structure
        grid_hash = np.random.rand(n, 2) * 0.1
        perturbed_v = v0.copy()
        for i in range(n):
            perturbed_v[3*i] += grid_hash[i, 0] * (1.0 / best_cols)
            perturbed_v[3*i+1] += grid_hash[i, 1] * (1.0 / best_rows)
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 600,
                "ftol": 1e-11,
                "gtol": 1e-11,
                "eps": 1e-9,
                "disp": False,
                "iprint": -1
            }
        )
    
    if not res.success:
        # Final fallback to original v0 with tighter bounds
        res = minimize(
            neg_sum_radii,
            v0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 600,
                "ftol": 1e-11,
                "gtol": 1e-11,
                "eps": 1e-9,
                "disp": False,
                "iprint": -1
            }
        )
    
    v = res.x if res.success else v0
    
    # Post-optimization enhancement with dynamic local reconfiguration 
    # Using geometric sensitivity scoring and radius boosting

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]
    # Validate for safety
    if not validate_packing(centers, radii)[0]:
        # If initial validation fails, reset and re-run
        res = minimize(
            neg_sum_radii,
            v0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 500,
                "ftol": 1e-11,
                "gtol": 1e-11,
                "eps": 1e-9,
                "disp": False,
                "iprint": -1
            }
        )
        v = res.x if res.success else v0
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
    
    # Dynamic geometric scoring for optimization guidance:
    # 1. Evaluate distances for each circle to all others
    # 2. Score based on min distance (high = less constrained)
    # 3. Assign a priority for expansion - circle with max score gets boost
    # 4. Also track circle constraints (i.e., tight packing, small min distance)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dists[i,j] = np.sqrt(dx*dx + dy*dy)
    # Compute min distance per circle and score
    min_dists = np.min(dists, axis=1)
    # For each circle, score is (min_dist / mean_dist) * (1 / radius)
    mean_dist = np.mean(min_dists)
    scores = min_dists / mean_dist * (np.sum(min_dists) / (np.sum(1.0 / (radii + 1e-10)) + 1e-10)) 
    # Normalize and select most constrained circle for expansion
    # For expansion, choose circles that are both least constrained (highest score) and smallest radius
    score_per_radius = scores / (radii + 1e-9)  # Weighted by radius
    # Find circle with highest score-per-radius ratio
    idx_to_expand = np.argmax(score_per_radius)
    
    # Perform a controlled expansion for the selected circle
    # Compute current total sum and possible expansion limit
    current_total = np.sum(radii)
    max_possible_radius = np.min( [1.0 - centers[i][0] - 1e-9, 1.0 - centers[i][1] - 1e-9, 
                            centers[i][0] - 1e-9, centers[i][1] - 1e-9] for i in range(n))
    avg_radius_per_circle = current_total / n
    expansion_amount = np.clip((max_possible_radius - radii[idx_to_expand]) * 0.75, 0, 0.0025)
    
    # Boost expansion by 10% if it is the least constrained
    if scores[idx_to_expand] > np.percentile(scores, 90):
        expansion_amount *= 1.1
    else:
        expansion_amount *= 0.9
    
    # Create a new radii array and adjust only the target circle
    new_r = radii.copy()
    new_r[idx_to_expand] = np.clip(radii[idx_to_expand] + expansion_amount, 1e-9, max_possible_radius)
    
    # Update the decision vector
    v_new = v.copy()
    v_new[2::3] = new_r
    
    # Re-optimizing with expanded radius
    res = minimize(
        neg_sum_radii,
        v_new,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 400,
            "ftol": 1e-11,
            "gtol": 1e-11,
            "eps": 1e-9,
            "disp": False,
            "iprint": -1
        }
    )
    
    v = res.x if res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)  # ensure no radii under threshold
    
    return centers, radii, float(radii.sum())