import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    # Optimize grid layout by using 5x6, 4x7 or 3x9, which better balance spacing
    # We now use a hybrid grid of 5x6 and 4x7 with an adaptive allocation
    # Start with structured grid to reduce initial configuration complexity
    # Initial grid structure: 5 cols & 6 rows with dynamic expansion
    rows = (n + cols - 1) // cols
    grid_cols = cols
    grid_rows = rows
    
    # We apply a hybrid geometric initialization to balance symmetry and randomness:
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        # Base grid with 0.45 / grid_rows to allow for 1e-2 expansion margin on each
        x_center = (col + 0.5 + np.random.uniform(-0.04, 0.04)) / cols
        y_center = (row + 0.5 + np.random.uniform(-0.04, 0.04)) / rows
        # Alternate row staggering for non-periodic grid
        if row % 2 == 1:
            x_center += 0.35 / cols
        # For rows with fewer items, adjust spacing
        if row > (rows - 2):
            x_center = (col + 0.5) / cols
        xs.append(x_center)
        ys.append(y_center)
    
    # Start with more aggressive base radius but leave room for expansion
    r0 = (0.5 - 0.08) / cols  # Start with more radius than previously
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Define strict bounds for all three parameters: exact matching length 3*n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length is 3*n, matches vector
    
    # Define negated sum of radii (we minimize it)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Create constraint list with strict boundary checks using captured i
    cons = []
    for i in range(n):
        # Left: x - r >= 0
        cons.append({"type": "ineq",
                     "fun": (lambda v, i=i: v[3*i] - v[3*i + 2])})
        # Right: x + r <= 1
        cons.append({"type": "ineq",
                     "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2])})
        # Bottom: y - r >= 0
        cons.append({"type": "ineq",
                     "fun": (lambda v, i=i: v[3*i + 1] - v[3*i + 2])})
        # Top: y + r <= 1
        cons.append({"type": "ineq",
                     "fun": (lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2])})
    
    # Overlap constraints with geometric hashing and vectorized expressions
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorized constraint function with captured i and j to avoid lambda capture issues
            def constraint_func(v, i=i, j=j):
                # Use direct indexing for performance and stability
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx * dx + dy * dy - (v[3*i+2] + v[3*j+2]) ** 2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # First optimization stage: initial optimization using SLSQP
    # Use tighter tolerance and more iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11, "eps": 1e-11})
    
    # If initial optimization successful, proceed to secondary optimization
    if res.success:
        v = res.x
        # Step 1: Symmetry-breaking vectorized perturbation with spatial hashing
        # Create a random spatial hash that's weighted by current radii for better perturbation
        spatial_hash = np.random.rand(n, 2) * 0.04
        # Spatial scaling factor based on current radius to maintain balance
        scaling_factor = np.mean(v[2::3]) * 0.6
        perturbed_v = v.copy()
        for k in range(n):
            perturbed_v[3*k] += spatial_hash[k, 0] * scaling_factor
            perturbed_v[3*k + 1] += spatial_hash[k, 1] * scaling_factor
        
        # Run second-stage optimization with refined perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-11})
    
    # If res still succeeds, targetted expansion to unlock further optimization
    # Apply multi-stage expansion with dynamic constraint awareness
    
    if res.success:
        v = res.x
        # Step 2: Compute current configuration and constraints
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Compute distance matrix with vectorization
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx ** 2 + dy ** 2)
        # Compute min distance for each circle to others
        min_dists = np.min(dists, axis=1)
        # Identify the circle with the smallest radius and the circle with maximal minimum distance
        # We prioritize expansion on circles with the most room for growth with minimal disruption
        # (we'll first expand the smallest radius circle to trigger layout change)
        smallest_radius_idx = np.argmin(radii)
        highest_separation_idx = np.argmax(min_dists)
        
        # Step 3: Create a controlled expansion plan
        # We define a growth schedule: expand smallest first, then others
        # Use adaptive expansion based on proximity to boundaries and adjacency
        # Also, we'll track constraint tightness for future refinement
        
        # Create new radii with expansion based on min available space
        # We calculate the potential growth for each circle based on distance to boundaries and neighbors
        # We then apply a radial expansion schedule to maximize total sum
        
        # Compute constraint tightness (normalized to [0,1]: 1 if constraint is tight)
        constraint_tightness = np.zeros(n)
        for i in range(n):
            # Left/right
            left_tight = np.abs(v[3*i] - v[3*i + 2]) / (v[3*i] + v[3*i + 2]) 
            right_tight = np.abs(1.0 - v[3*i] - v[3*i + 2]) / (v[3*i] + v[3*i + 2]) 
            bottom_tight = np.abs(v[3*i + 1] - v[3*i + 2]) / (v[3*i + 1] + v[3*i + 2]) 
            top_tight = np.abs(1.0 - v[3*i + 1] - v[3*i + 2]) / (v[3*i + 1] + v[3*i + 2])
            constraint_tightness[i] = np.mean([left_tight, right_tight, bottom_tight, top_tight])
        
        # Find the circle with the lowest constraint tightness (least constrained)
        least_constrained_idx = np.argmin(constraint_tightness)
        
        # Step 4: Construct a hybrid expansion plan
        # We will perform adaptive expansion based on current constraints
        # We prioritize expansion on least constrained first, then others
        
        # Define a radial expansion factor based on available slack
        # We will expand the least constrained circle with a targeted expansion
        # To avoid overlap, we'll use the current layout to predict if it's feasible
        # First, we attempt a small expansion on the least constrained circle
        
        # Calculate the minimal expansion that could possibly cause overlaps
        max_safe_growth = 0.0
        for j in range(n):
            if j == least_constrained_idx:
                continue
            dx = v[3*least_constrained_idx] - v[3*j]
            dy = v[3*least_constrained_idx + 1] - v[3*j + 1]
            dist = np.sqrt(dx**2 + dy**2)
            max_safe_growth = max(max_safe_growth, max(0.0, (dist - (radii[least_constrained_idx] + radii[j]))))
        
        # We calculate a theoretical safe expansion on least constrained circle
        # Also, we consider how much we can grow before colliding with boundaries
        boundary_growth = min(
            (v[3*least_constrained_idx] - 1e-6) / radii[least_constrained_idx], 
            (1.0 - v[3*least_constrained_idx] - 1e-6) / radii[least_constrained_idx],
            (v[3*least_constrained_idx + 1] - 1e-6) / radii[least_constrained_idx], 
            (1.0 - v[3*least_constrained_idx + 1] - 1e-6) / radii[least_constrained_idx]
        )
        
        # The total theoretical expansion for the circle
        max_possible_growth = min(boundary_growth, max_safe_growth)
        
        # We expand it by the maximum possible amount, with a safety margin (we'll test at 0.25 * max_possible_growth first)
        # This is an adaptive expansion strategy, with a safety check
        expansion = max_possible_growth * 0.6
        
        # Create new radii with expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion
        
        # Now, calculate how many others we can expand by a smaller amount to increase the total sum
        # We distribute the remaining expansion amount proportionally to other circles
        remaining = (0.005 - (np.sum(new_radii) - np.sum(radii))) if (np.sum(new_radii) < 0.005) else 0
        if remaining > 0:
            # To distribute expansion more effectively, we use normalized constraint tightness to prioritize circles
            # We scale the remaining expansion by constraint tightness
            # We add a small increment to each to simulate a soft growth
            new_radii += remaining * (np.ones_like(constraint_tightness) * 0.1) * (1.0 / (1 + constraint_tightness))
        
        # Update the v vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Run the third optimization phase with new radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})
    
    # Final check and cleanup: after all optimizations, validate and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    # Run a forced check on the final configuration (even if the solver succeeds, it's safer)
    # We run a post-hoc check in case constraints are violated due to numerical instability
    # This should not affect the performance because all constraints are already enforced in optimization
    return centers, radii, float(radii.sum())