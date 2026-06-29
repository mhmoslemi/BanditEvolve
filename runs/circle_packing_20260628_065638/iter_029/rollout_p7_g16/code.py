import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with a hybrid of geometric clustering, perturbed grid, and adaptive staggering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce non-uniform perturbation to break symmetry and simulate natural clustering
        row_weight = 1.0 - 0.7 * (row / (rows - 1)) if rows > 1 else 1.0
        col_weight = 1.0 - 0.6 * (col / (cols - 1)) if cols > 1 else 1.0
        x_perturb = np.random.uniform(-0.08 * col_weight, 0.08 * col_weight)
        y_perturb = np.random.uniform(-0.08 * row_weight, 0.08 * row_weight)
        x = x_center + x_perturb
        y = y_center + y_perturb
        # Stagger alternate rows with adaptive offset to increase packing density
        if row % 2 == 1:
            x += 0.5 / cols * np.random.uniform(0.5, 1.2)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i
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
    
    # Vectorized overlap constraints with adaptive scaling to reduce numerical noise
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # First phase: global optimization with aggressive parameterization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3500, "ftol": 1e-11, "eps": 1e-12})
    
    # Phase 2: dynamic dissection of topologically significant pairs using constraint decomposition
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute pairwise distances and identify topologically active pairs
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify topological pivot pairs: those with minimal relative distance to their cluster
        # Define cluster affinity by sum of squared distances for each circle
        cluster_affinity = np.zeros(n)
        for i in range(n):
            cluster_affinity[i] = np.sum((dists[i, :] - radii[i]) ** 2)
        
        # Select two most dynamically interacting circles (not just smallest radius)
        # Use a hybrid metric combining direct distance and cluster influence
        interaction_metric = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                interaction_metric[i] += 0.8 * (dists[i, j] - (radii[i] + radii[j])) ** 2
                interaction_metric[j] += 0.8 * (dists[i, j] - (radii[i] + radii[j])) ** 2
        # Find top two circles with highest interaction metric (most "active")
        top_pair = np.argsort(interaction_metric)[-2:]
        
        # Phase 3: Constraint-driven reconfiguration of top_pair
        # Temporarily remove constraints between top_pair
        # and re-optimize to force reconfiguration
        temp_constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i != top_pair[0] or j != top_pair[1]) and (i != top_pair[1] or j != top_pair[0]):
                    temp_constraints.append({"type": "ineq", 
                                             "fun": (lambda v, i=i, j=j: 
                                                     (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                                     - (v[3*i+2] + v[3*j+2])**2)})
        
        # Add new constraints to enforce reconfiguration
        # Add constraint to force minimal separation between top_pair
        temp_constraints.append({"type": "ineq",
                                 "fun": lambda v: (v[3*top_pair[0]] - v[3*top_pair[1]])**2 + (v[3*top_pair[0]+1] - v[3*top_pair[1]+1])**2 - (radii[top_pair[0]] + radii[top_pair[1]] + 0.01)**2})
        
        # Re-optimization with partial constraint set
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=temp_constraints, 
                       options={"maxiter": 1000, "ftol": 1e-11, "eps": 1e-12})
    
    # Phase 4: Topologically driven expansion with controlled constraint reordering
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Compute all pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # Identify the least constrained circle by checking minimum required expansion
        # Create constraint violation map
        violation_map = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                violation = (dists[i, j] - (radii[i] + radii[j])) ** 2
                if violation < 1e-10:
                    violation = 1e-10
                violation_map[i, j] = violation
                violation_map[j, i] = violation
        
        # Calculate the "leverage" of expansion for each circle (minimal constraint violation)
        leverages = np.zeros(n)
        for i in range(n):
            leverages[i] = np.mean(violation_map[i, :])
        
        # Select the circle with minimal constraint leverage for targeted expansion
        least_constrained_idx = np.argmin(leverages)
        # Create expansion vector with controlled growth
        # We aim to increase radii by 0.0075 while maintaining topology
        target_total = np.sum(radii) + 0.0075
        expansion_amount = target_total / n

        # Create new_radius vector by distributing expansion while respecting constraints
        # Ensure expansion doesn't create new overlaps
        # Temporarily add new constraint for expansion direction
        temp_constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                temp_constraints.append({"type": "ineq", 
                                        "fun": (lambda v, i=i, j=j: 
                                            (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                            - (v[3*i+2] + v[3*j+2])**2)})
        
        # Add new constraint to force a minimal expansion direction for the least constrained
        # Assume least constrained circle is at a lower "topology height" in the grid
        temp_constraints.append({"type": "ineq", 
                                 "fun": lambda v: (v[3*least_constrained_idx+1] - (1 - 0.1)) ** 2 - (v[3*least_constrained_idx+2]**2)})

        # Create new radii with distributed expansion and minimal overlap
        new_radii = radii * 1.0
        expansion_amount = (target_total - np.sum(radii)) / (n-1) if n > 1 else 0
        # Increase only the least constrained circle
        new_radii[least_constrained_idx] += expansion_amount * 1.2
        # Add a secondary expansion to neighboring clusters to preserve topology
        neighbor_distances = np.zeros(n)
        for i in range(n):
            if i != least_constrained_idx:
                dx = centers[i, 0] - centers[least_constrained_idx, 0]
                dy = centers[i, 1] - centers[least_constrained_idx, 1]
                neighbor_distances[i] = dx**2 + dy**2
        neighbor_radii = np.sort(neighbor_distances)[::-1]  # Sort by proximity
        max_neighbor = neighbor_radii[0]
        second_neighbor = neighbor_radii[1]
        # Increase neighbors' radii in relation to their proximity to the least constrained
        for i in range(n):
            if i != least_constrained_idx:
                ratio = neighbor_distances[i] / max_neighbor if max_neighbor > 1e-8 else 1.0
                if neighbor_distances[i] < second_neighbor or ratio < 0.2:
                    new_radii[i] += expansion_amount * (0.5 + 0.5 * (1 - ratio))
        
        # Create new decision vector with distributed radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with new constraint set and new radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=temp_constraints, 
                       options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-12})
    
    # Final post-validation refinement
    if res.success:
        v = res.x
        # Ensure all coordinates are in bounds
        for i in range(n):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            if x - r < -1e-12:
                v[3*i] = max(0.0, x)
            if x + r > 1 + 1e-12:
                v[3*i] = min(1.0, x)
            if y - r < -1e-12:
                v[3*i+1] = max(0.0, y)
            if y + r > 1 + 1e-12:
                v[3*i+1] = min(1.0, y)
            # Ensure radii within bounds
            v[3*i+2] = np.clip(v[3*i+2], 1e-6, 0.5)
        # Re-optimize with bounds applied and final constraint check
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, 
                       options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())