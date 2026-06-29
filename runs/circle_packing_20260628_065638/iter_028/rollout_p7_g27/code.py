import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Generate optimized initial positions with hierarchical spatial organization
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Base offset to avoid clustering and promote separation
        x_offset_base = np.random.uniform(-0.04, 0.04)
        y_offset_base = np.random.uniform(-0.04, 0.04)
        # Additional offset to create non-uniformity
        row_weight = np.sin(np.pi * row / rows) + 1
        col_weight = np.cos(np.pi * col / cols) + 1
        x_offset_add = np.random.uniform(-0.01, 0.01) * row_weight
        y_offset_add = np.random.uniform(-0.01, 0.01) * col_weight
        # Apply row staggering for non-grid layout
        if row % 2 == 1:
            x_center += 0.5 / cols * 0.95 
        x = x_center + x_offset_base + x_offset_add
        y = y_center + y_offset_base + y_offset_add
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

    # Vectorized constraints for boundaries with captured i
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
    
    # Vectorized overlap constraints with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations, tighter tolerance, and gradient approximation
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-8})

    # First-phase spatial reconfiguration: geometric hashing with enhanced perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Build directional sensitivity map for spatial perturbation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        dists[dists < 1e-12] = np.inf  # Avoid zero-distance issues
        
        # Compute directional sensitivity matrix for enhanced perturbation
        directional_sensitivity = np.zeros((n, n, 2))
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = dists[i,j]
                    dx_ = dx[i,j]
                    dy_ = dy[i,j]
                    if d != np.inf:
                        directional_sensitivity[i,j,0] = dx_ / d
                        directional_sensitivity[i,j,1] = dy_ / d
        
        # Generate directional perturbation using sensitivity map and randomized seed
        np.random.seed(42)  # Ensure deterministic spatial hashing
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Apply perturbation scaled by sensitivity to surrounding circles
            direction = np.zeros(2)
            weight = 0.0
            for j in range(n):
                if i != j:
                    d = dists[i,j]
                    if d < 0.2:
                        dir_comp = directional_sensitivity[i,j]
                        direction += dir_comp * (1.0 - d / 0.2) ** 2
                        weight += 1.0 - d / 0.2
            direction /= weight if weight > 0 else 1.0
            perturbed_v[3*i] += spatial_hash[i,0] * (radii[i] / np.mean(radii)) * (1.0 + 0.2 * np.random.rand())
            perturbed_v[3*i+1] += spatial_hash[i,1] * (radii[i] / np.mean(radii)) * (1.0 + 0.2 * np.random.rand())
            perturbed_v[3*i+2] += np.random.uniform(-0.002, 0.002)
        
        # Re-evaluate with enhanced spatially aware configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-8})

    # Second-phase: geometric dissection and topological transformation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute adjacency-based topological signature matrix
        dx_adj = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy_adj = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists_adj = np.sqrt(dx_adj**2 + dy_adj**2)
        dists_adj[dists_adj < 1e-12] = np.inf
        
        # Identify top two interacting circles with weighted adjacency matrix
        adjacency_weights = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    if dists_adj[i,j] < 1.5 * np.max(radii):
                        adjacency_weights[i,j] = np.exp(-np.log(10) * (dists_adj[i,j] / (radii[i] + radii[j])))
                    else:
                        adjacency_weights[i,j] = 0.0
        interaction = np.sum(adjacency_weights, axis=1)
        top_idx = np.argsort(interaction)[-2:]  # Get top two
        second_idx = np.argsort(interaction)[-3]  # Get third for control
        
        # Disjoint reconfiguration for top two circles
        perturbed_v = v.copy()
        for i in top_idx:
            # Apply directional perturbation and radius expansion
            direction = np.array([0.0, 0.0])
            weight = 0.0
            for j in range(n):
                if i != j:
                    d = dists_adj[i,j]
                    if d < 0.4:
                        dir_comp = directional_sensitivity[i,j]
                        direction += dir_comp * (1.0 - d / 0.4) ** 2
                        weight += 1.0 - d / 0.4
            if weight > 0:
                direction /= weight
            # Apply large spatial adjustment and radius expansion for top circles
            perturbed_v[3*i] += direction[0] * 0.06 * (radii[i] / np.mean(radii)) * 1.2
            perturbed_v[3*i+1] += direction[1] * 0.06 * (radii[i] / np.mean(radii)) * 1.2
            perturbed_v[3*i+2] += np.random.uniform(0.002, 0.004)
        
        # Re-evaluate with reconfigured top circles
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-8})

    # Third-phase: constrained topological reconfiguration with hybrid radius adjustment
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute minimal isolation score for radius expansion
        dx_min = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy_min = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists_min = np.sqrt(dx_min**2 + dy_min**2)
        dists_min[dists_min < 1e-12] = np.inf
        
        # Identify most isolated circle
        isolation = np.min(dists_min, axis=1)
        isolated_idx = np.argmax(isolation)
        
        # Compute radius expansion budget based on topological constraints
        base_radius = np.mean(radii)
        radius_budget = 0.005 * (1.0 + 0.2 * np.random.rand())  # Add stochasticity for expansion
        expansion_weights = np.zeros(n)
        expansion_weights[isolated_idx] = 1.3  # Higher expansion ratio for isolated circle
        
        # Apply gradual expansion while preserving non-overlap
        new_radius_values = radii.copy()
        while True:
            # Compute expanded configuration
            new_radius_values = np.clip(new_radius_values, 1e-6, 0.5)
            expanded_v = v.copy()
            expanded_v[2::3] = new_radius_values
            
            # Compute all distances again and validate
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)
            dists[dists < 1e-12] = np.inf
            
            # Check for overlap
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    if dists[i,j] < new_radius_values[i] + new_radius_values[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion
                expansion_weights = np.clip(expansion_weights * 0.95, 0.1, 1.3)
        
        # Apply new radius values
        v[2::3] = new_radius_values
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-8})

    # Final optimization with adaptive constraint tightening
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Final radius refinement using directional gradient approach
        dx_adj = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy_adj = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists_adj = np.sqrt(dx_adj**2 + dy_adj**2)
        dists_adj[dists_adj < 1e-12] = np.inf
        
        for i in range(n):
            # Compute directional influence for radii
            influence = np.zeros(3)
            for j in range(n):
                if i != j:
                    d = dists_adj[i,j]
                    if d < 0.5:
                        influence[0] += (d - radii[i] - radii[j]) * (d - 0.5)**2
                        influence[1] += (d - radii[i] - radii[j]) * (d - 0.5)**2
                        influence[2] += (d - radii[i] - radii[j]) * (d - 0.5)**2
            # Apply directional gradient to radii
            if np.abs(influence[0]) > 1e-8 or np.abs(influence[1]) > 1e-8 or np.abs(influence[2]) > 1e-8:
                v[3*i + 2] += (influence[0] + influence[1] + influence[2]) * 1e-4
        
        # Final evaluation
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())