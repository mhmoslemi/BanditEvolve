import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with optimized seed-based spatial perturbations
    seed = np.random.randint(1, 1000000)
    np.random.seed(seed)
    
    # Spatial arrangement with optimized staggered grid geometry
    xs = []
    ys = []
    
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x_center = (col + 0.5) / cols
        base_y_center = (row + 0.5) / rows
        
        # Apply multi-level spatial perturbation
        jitter = np.random.uniform(-0.01, 0.01, size=2)
        x = base_x_center + jitter[0] + np.sin(row) * 0.01
        y = base_y_center + jitter[1] + np.cos(col) * 0.01
        
        # Stagger even/odd rows with geometrically increasing offset
        if row % 2 == 1:
            x += 0.5 / cols * (1 - 0.9 ** row)  # Decay factor to prevent over-staggering
        
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with optimized geometric scaling
    base_radius = 0.4 / cols * (1 - (0.95 ** (1 / rows)))  # Non-linear radius scaling for even spread
    r0 = base_radius - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Create tight bounds matching exactly the decision vector length 3n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries, 26 circles

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize sum of radii

    # Define advanced constraints with improved numerical stability
    cons = []

    # Bound constraints (each has its own 4 constraints for circle)
    for i in range(n):
        # x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})

        # 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})

        # y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

        # 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Overlap constraints (with enhanced numerical stability and vectorization)
    for i in range(n):
        for j in range(i + 1, n):
            # Use vectorized lambda closure for numerical stability
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2
            })

    # Initial optimization with enhanced tolerances, multi-phase strategy
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 2400, 
                       "ftol": 1e-12,
                       "eps": 1e-8,
                       "disp": False
                   })
    
    # Symmetric reconfiguration with adaptive perturbation and gradient boosting
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Adaptive perturbation based on spatial distribution and density
        perturbation_factor = 0.1 + (np.std(np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))[:, 1:])) * 0.05
        
        # Create perturbation with exponential decay based on distance
        perturbation = np.random.rand(n, 2) * 0.1 * (1 - np.exp(-0.6 * np.mean(radii)))
        perturbed_v = v.copy()
        for i in range(n):
            x_perturb = perturbation[i, 0] * (1 + 0.2 * np.sin(5 * centers[i, 0]))
            y_perturb = perturbation[i, 1] * (1 + 0.2 * np.cos(5 * centers[i, 1]))
            perturbed_v[3*i] += x_perturb
            perturbed_v[3*i+1] += y_perturb
        
        # Perform reconfiguration with enhanced gradient handling
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 200, 
                           "ftol": 1e-12,
                           "eps": 1e-8,
                           "disp": False
                       })
    
    # Targeted optimization on least constrained circle with spatial awareness
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate minimum distances to other circles for each circle
        distances = np.zeros(n)
        for i in range(n):
            min_dist = np.inf
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < min_dist:
                        min_dist = dist
            distances[i] = min_dist
        
        # Select the circle with the most free space (least constraining) to expand
        least_constrained_idx = np.argmax(distances)
        other_radii_sum = np.sum(radii[np.arange(n) != least_constrained_idx])
        
        # Calculate maximum possible expansion
        max_possible_expansion = (distances[least_constrained_idx] - radii[least_constrained_idx] - 1e-8) * 0.8
        max_possible_sum_increase = (1.25 * max_possible_expansion) * other_radii_sum
        
        # Target growth to achieve a significant boost while maintaining stability
        target_total_increase = max_possible_sum_increase * 0.8
        expansion_factor = target_total_increase / (np.sum(radii) + 1e-12)
        
        # Generate optimized expansion vector
        expansion_vec = np.zeros(n)
        expansion_vec[least_constrained_idx] = expansion_factor * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                expansion_vec[i] = expansion_factor * (1.0 + 0.1 * np.random.rand()) * (distances[i] / np.max(distances))
        
        # Apply expansion vector
        expanded_v = v.copy()
        expanded_v[2::3] = radii + expansion_vec
        
        # Final refinement with tight convergence for optimal solution
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 250, 
                           "ftol": 1e-12,
                           "eps": 1e-8,
                           "disp": False
                       })

    # Final check and clipping to maintain physical feasibility
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())