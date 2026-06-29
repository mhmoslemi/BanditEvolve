import numpy as np

def run_packing():
    n = 26
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
        
        # Apply non-uniform perturbation with adaptive range based on row proximity
        x = x_center + np.random.uniform(-0.045 * (1 - row / rows), 0.045 * (1 - row / rows))
        y = y_center + np.random.uniform(-0.045 * (1 - row / rows), 0.045 * (1 - row / rows))
        
        # Alternate row shifting with diminishing impact
        if row % 2 == 1:
            x += 0.5 / cols * (1 - 0.5 * (row / rows))
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries for position and radius

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with efficient lambda capturing
    cons = []
    for i in range(n):
        # Left boundary: x[i] - r[i] >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary: x[i] + r[i] <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary: y[i] - r[i] >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary: y[i] + r[i] <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})

    # Vectorized overlap constraints with efficient lambda capturing and squared distance
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Primary optimization with aggressive tolerance and max iter
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-11})

    # Radical geometric reconfiguration with spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hashing with nonlinear scaling for spatial perturbation
        spatial_hash = np.random.rand(n, 2) * (0.05 + 0.02 * np.sin(np.pi * np.arange(n) / 10))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))**0.8
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))**0.8
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-12})

    # Topological reordering with constrained radius expansion on smallest non-zero circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find smallest non-zero radius for targeted expansion
        smallest_radius_idx = np.argmin(radii[radii > 1e-6])
        
        # Adaptive expansion heuristic with dynamic growth multiplier
        base_growth = 0.008
        current_total = np.sum(radii)
        potential = (np.min(dists[np.triu_indices(n, 1)]) - np.min(radii)) / (np.max(radii) + 1e-6)
        expansion_factor = base_growth * (1 + 0.5 * potential) / (n - 1)
        
        # Create expansion vector with prioritized smallest radius expansion
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != smallest_radius_idx:
                new_radii[i] += expansion_factor * np.random.uniform(0.8, 1.1)
        
        # Iterative refinement with local validation and adaptive contraction
        for _ in range(3):
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances between circles
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    # Use squared distance to avoid sqrt for faster calculation
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < (new_radii[i] + new_radii[j]) ** 2 - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion proportionally based on overlap severity
                overlap_intensity = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                        # Use squared distance to avoid sqrt for faster calculation
                        dist_sq = dx*dx + dy*dy
                        if dist_sq < (new_radii[i] + new_radii[j]) ** 2 - 1e-12:
                            overlap_intensity += 1 / (new_radii[i] + new_radii[j])
                new_radii = radii + (new_radii - radii) * (1 - 0.7 * overlap_intensity / (n * (n - 1)))
        
        # Final refinement with optimized decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())