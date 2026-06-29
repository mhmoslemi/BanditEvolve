import numpy as np
import warnings

def run_packing():
    n = 26
    # Use an optimized hexagonal grid with adaptive perturbation and dual-phase spatial constraints
    
    # Optimal grid dimensions for 26 circles with hexagonal packing logic
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x_rand = np.random.uniform(-0.06, 0.06)
        y_rand = np.random.uniform(-0.04, 0.04)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x_center = (col + 0.5 + 0.5 / cols) / cols
        x = x_center + x_rand
        y = y_center + y_rand
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation with adaptive scaling based on grid density
    # For optimized hex grid, radius ~ 0.15 / sqrt(3) per column
    r0 = 0.42 / cols - 1e-3  # Slightly higher than base to allow expansion
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bounds for 3n variables
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize sum by minimizing negative
    
    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with optimized closure capture
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with capture to avoid closure issues in nested loops
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})
    
    # First optimization phase: base layout with initial radii
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-10,
                                             "disp": False})
    
    # If optimization was successful, perform reconfiguration
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]
    
    # Apply targeted geometric dissection on 2 most interactively constrained circles
    # Step 1: Identify most interacting pair using vectorized distance
    dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
    dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    # Identify pair with most interactions = min distance across all non-adjacent pairs
    # Use inverse of distance as interaction strength
    interaction_strength = 1.0 / (dists + 1e-6) # avoid divide by zero
    # Exclude i=j (diagonal) to not penalize same circle
    np.fill_diagonal(interaction_strength, 0)
    # Find indices of top 2 interacting pairs
    top_two_pairs = np.argpartition(interaction_strength, -2, axis=None)[-2:]
    i1, j1 = divmod(top_two_pairs[0], n)
    i2, j2 = divmod(top_two_pairs[1], n)
    
    # Step 2: Isolate and reconfigure the two most interacting circles
    # Apply dynamic directional displacement to create spatial gap
    # Apply gradient descent optimization on just the two circles, with all other circles fixed
    # Set other circles as fixed, only optimize the two circles
    # Create a reduced problem where 2 circles are variables, the rest are fixed
    # Initialize v_sub with the 2 optimized circles
    v_sub = v.copy()
    v_sub[3*i1] = centers[i1, 0] + np.random.uniform(-0.04, 0.04)  # perturb slightly
    v_sub[3*i1+1] = centers[i1, 1] + np.random.uniform(-0.04, 0.04)
    v_sub[3*i2] = centers[i2, 0] + np.random.uniform(-0.04, 0.04)
    v_sub[3*i2+1] = centers[i2, 1] + np.random.uniform(-0.04, 0.04)
    v_sub[3*i1+2] = radii[i1] + np.random.uniform(-0.05, 0.05)
    v_sub[3*i2+2] = radii[i2] + np.random.uniform(-0.05, 0.05)
    v_sub[3::3] = np.nan  # freeze other radii
    v_sub[3*i1+2::3] = np.nan  # freeze radii of other circles
    
    # Create bounds for the reconfiguration
    bounds_reconfig = []
    for k in range(n):
        if k == i1 or k == i2:
            # These are variables
            bounds_reconfig += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
        else:
            # Freeze these to original values
            bounds_reconfig += [(v[3*k], v[3*k]), (v[3*k+1], v[3*k+1]), (v[3*k+2], v[3*k+2])]
    
    # Create new constraint list with frozen positions except for two circles
    # All interaction constraints still active, but with two circles as variables
    cons_reconfig = []
    for k in range(n):
        # All boundaries still active
        for c in range(4):
            cons_reconfig.append({"type": "ineq", "fun": (
                lambda v, k=k, c=c: 
                (1.0 - v[3*k] - v[3*k+2]) if c == 0 else 
                (v[3*k] - v[3*k+2]) if c == 1 else 
                (1.0 - v[3*k+1] - v[3*k+2]) if c == 2 else 
                (v[3*k+1] - v[3*k+2]) if c == 3 else 0.0
            )})
    
    # Only check overlaps that are not between the two optimized circles
    # So for i in 0 to n-1, j in i+1 to n-1:
    # if both i and j are not the two fixed circles, include the constraint
    for i in range(n):
        for j in range(i+1, n):
            if i != i1 and i != i2 and j != i1 and j != i2:
                cons_reconfig.append({"type": "ineq", "fun": (
                    lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                        - (v[3*i+2] + v[3*j+2])**2
                )})
    
    # Now run optimization only on the two circles
    res_reconfig = minimize(neg_sum_radii, v_sub, method="L-BFGS-B", bounds=bounds_reconfig,
                            constraints=cons_reconfig, options={"maxiter": 1500, "ftol": 1e-12,
                                                                "eps": 1e-10, "disp": False})
    
    # Apply the reconfirmed positions from reconfiguration
    if res_reconfig.success:
        v = res_reconfig.x.copy()
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
    
    # Step 3: Identify least constrained circle for targeted expansion
    # Calculate pairwise distances matrix
    dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
    dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
    dists = np.sqrt(dx**2 + dy**2)
    min_dists = np.min(dists, axis=1)
    least_constrained_idx = np.argmax(min_dists)  # Most distance to other circles
    
    # Apply controlled expansion strategy on least constrained
    # Calculate current total for growth factor
    current_total = np.sum(radii)
    # Dynamic expansion based on spatial configuration and current radius
    expansion_factor = 0.0072
    # Targeted expansion for least constrained circle
    # Apply exponential growth with spatial perturbation
    expansion = expansion_factor * (1 + 0.1 * np.random.rand())  # Stochastic expansion
    if radii[least_constrained_idx] + expansion < 0.4:
        expansion *= 0.5
    new_radii = radii.copy()
    new_radii[least_constrained_idx] += expansion
    
    # Apply expansion to adjacent circles based on spatial proximity
    for k in range(n):
        if k == least_constrained_idx:
            continue
        # Use directional bias from spatial configuration
        direction = (centers[k] - centers[least_constrained_idx])
        unit_dir = direction / np.linalg.norm(direction + 1e-6)
        expansion = expansion_factor * (1.0 + 0.3 * np.random.rand())
        # If circle is nearby
        if np.linalg.norm(direction) < 0.25:
            expansion *= 1.5
        # Add expansion but constrained by maximum allowed radius
        if new_radii[k] + expansion < 0.4:
            new_radii[k] += expansion
        else:
            # Add proportional expansion to avoid overflow
            new_radii[k] += (0.4 - radii[k])
    
    # Apply expansion with constraint validation
    while True:
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
        
        # Validate expanded configuration
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                dist = np.sqrt(dx_exp**2 + dy_exp**2)
                if dist < new_radii[i] + new_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            break
        else:
            # If invalid, decrease expansion slightly for all circles
            new_radii = radii + (new_radii - radii) * 0.98
    
    # Update decision vector with reconfigured positions and expanded radii
    v_new = v.copy()
    v_new[2::3] = new_radii
    
    # Final optimization phase with enhanced spatial awareness and adaptive regularization
    res_final = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                        constraints=cons, options={"maxiter": 800, "ftol": 1e-11,
                                                  "eps": 1e-10, "disp": False})
    
    # Return the final configuration
    v = res_final.x if res_final.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid, reason = validate_packing(centers, radii)
    
    if not valid:
        # Fallback to the last successful configuration
        centers, radii, _ = run_packing()
    
    return centers, radii, float(radii.sum())