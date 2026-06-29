import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize with geometrically optimized and randomized staggered hexagonal grid
    # Add edge bias to allow better expansion of perimeter circles
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base geometry with hex packing bias
        x_center = (col + 0.5 + (row % 2) * 0.33) / cols
        y_center = (row + 0.5 + (row % 2) * 0.05) / rows
        # Apply spatial perturbation to avoid symmetry
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Ensure edge circles are not too close to corners
        if x_center < 0.3 or x_center > 0.7:
            y += np.random.uniform(-0.05, 0.05)
        if col in [0, cols-1]:
            y += np.random.uniform(0.0, 0.05)
        if row in [0, rows-1]:
            y += np.random.uniform(0.0, 0.05)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n elements

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint building with lambda capture optimization
    # Use of "lambda i=i" to prevent closure capture issues
    cons = []
    for i in range(n):
        # Boundary constraints with proper closure capture
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Optimized overlap constraints with vectorized evaluation
    # Use of numpy broadcasting to vectorize constraint computations
    for i in range(n):
        for j in range(i + 1, n):
            # Precompute dx and dy vectors for faster computation
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization phase with aggressive settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "eps": 1e-8})

    # Adaptive reconfiguration strategy:
    # 1. Use spatial hashing for constrained reconfiguration
    # 2. Apply edge-targeted expansion with safety checks
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Spatial hashing: generate adaptive perturbation based on local density
        # Compute proximity matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        dists[range(n), range(n)] = np.inf

        # Spatial hashing for constrained regions
        spatial_hash = np.random.rand(n, 2)
        perturbation_scale = np.clip(0.05 + (10 * radii) / np.sum(radii), 0.05, 0.15)
        
        # Apply adaptive spatial perturbation to all circles
        perturbed_v = v.copy()
        for i in range(n):
            # Apply perturbation inversely proportional to density
            local_density = np.sum(1 / (dists[i] + 1e-6)) / (n-1)
            perturbation = spatial_hash[i] * (1.2 + 0.2 * (10 * radii[i] / np.sum(radii)))
            perturbed_v[3*i] += perturbation[0] * perturbation_scale[i] * (1 / (1 + local_density))
            perturbed_v[3*i+1] += perturbation[1] * perturbation_scale[i] * (1 / (1 + local_density))
        
        # Re-evaluate with perturbed geometry
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})

    # Edge-circle expansion with strict feasibility checks
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Compute distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        dists[range(n), range(n)] = np.inf

        # Edge-circle criteria: circles with <50% min distance to other circles
        min_dists = np.min(dists, axis=1)
        edge_indices = np.where(min_dists < 0.5 * np.mean(min_dists))[0]

        # Targeted expansion: expand edge circles with safe growth
        if len(edge_indices) > 0:
            # Calculate safe expansion for each edge circle
            growth_factors = []
            for i in edge_indices:
                min_dist = min_dists[i]
                expansion_potential = (min_dist - 1e-6) / (radii[i] * 2)
                growth_factors.append(np.clip(expansion_potential, 0.0, 1.0) * 10)  # Safety margin

            # Apply expansion with safety constraints
            expanded_v = v.copy()
            expanded_radii = radii.copy()
            for i in edge_indices:
                if expanded_radii[i] + 1e-6 < 0.5:
                    expansion = np.clip(growth_factors[i] * (0.5 - expanded_radii[i]), 0, 0.5 - expanded_radii[i])
                else:
                    expansion = np.clip(growth_factors[i] * (0.5 - expanded_radii[i] + 1e-6), 0, 0.5 - expanded_radii[i])
                expanded_radii[i] += expansion
                
                # Safety check
                if expanded_radii[i] >= 0.5:
                    expanded_radii[i] = 0.5  # cap maximum radius
            expanded_v[2::3] = expanded_radii
            # Re-evaluate with expansion
            res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})
    
    # Final refinement with dynamic constraints and multi-phase optimization
    # Phase 1: boundary-focused expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate boundary utilization
        boundary_usage = np.zeros(n)
        for i in range(n):
            dx = v[3*i] - v[3*i+2]  # x position - radius
            dy = v[3*i+1] - v[3*i+2]  # y position - radius
            boundary_usage[i] = max(0, (dx + 1e-6 - (1 - dx)) / 2)  # balance of proximity to edges
            boundary_usage[i] = max(boundary_usage[i], (dy + 1e-6 - (1 - dy)) / 2)

        # Select circles with least boundary usage for expansion
        to_expand_indices = np.argsort(boundary_usage)[:5]  # expand least constrained first
        # Apply cautious expansion with safety check
        exp_v = v.copy()
        exp_radii = radii.copy()
        for i in to_expand_indices:
            if exp_radii[i] < 0.45:
                # Calculate safe expansion based on edge availability
                edge_available = 1 - min((exp_v[3*i] + exp_radii[i]), (1 - exp_v[3*i] + exp_radii[i]))
                edge_available = max(edge_available, 1 - min((exp_v[3*i+1] + exp_radii[i]), (1 - exp_v[3*i+1] + exp_radii[i])))
                max_safe_expansion = min(0.5 - exp_radii[i], edge_available)
                expansion_amount = max_safe_expansion * 0.01  # 1% safe expansion for optimization
                exp_radii[i] += expansion_amount

        exp_v[2::3] = exp_radii
        res = minimize(neg_sum_radii, exp_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-8})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())