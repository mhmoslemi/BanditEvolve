import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions using non-uniform spatial hashing and cluster-aware distribution
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5 + np.random.rand() * 0.3) / cols
        y_center = (row + 0.5 + np.random.rand() * 0.3) / rows
        # Add geometric noise for asymmetric clustering
        x = x_center + np.random.normal(0, 0.02)
        y = y_center + np.random.normal(0, 0.02)
        # Adjust for staggered spatial hashing
        if row % 2 == 1:
            x += 0.25 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure proper bounds alignment with 3*n length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints using broadcasting and geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerances and enhanced solver options
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "maxls": 200})
    
    # Geometric hashing transformation: randomized re-encoding of spatial coordinates
    if res.success:
        v = res.x
        # Compute centers and radii for validation
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Apply geometric hashing transformation with probabilistic spatial reconfiguration
        # Re-seed for deterministic hash in this context
        np.random.seed(42)
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration and enforce exact non-overlap
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "maxls": 200})
    
    # Targeted expansion of smallest radius while enforcing strict non-overlap
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation and isolation metric
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        
        # Find the circle with smallest non-zero radius
        non_zero_radii = np.clip(radii, 1e-6, None)
        smallest_radius_idx = np.argmin(non_zero_radii)
        smallest_radius = non_zero_radii[smallest_radius_idx]
        
        # Generate expansion vector based on isolation and non-overlap constraints
        new_radii = radii.copy()
        target_total_sum = np.sum(radii) + 0.008
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Apply controlled expansion starting from smallest radius
        new_radii[smallest_radius_idx] = smallest_radius + expansion_factor * 1.1
        for i in range(n):
            if i != smallest_radius_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.05 * np.random.rand())
        
        # Create a perturbation vector to force layout reconfiguration
        perturbation = np.random.rand(n, 3) * 0.03
        v_new = v.copy()
        v_new[2::3] = new_radii
        for i in range(n):
            v_new[3*i] += perturbation[i, 0]
            v_new[3*i+1] += perturbation[i, 1]
            v_new[3*i+2] += perturbation[i, 2]
        
        # Re-evaluate with new radii and spatial configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "maxls": 200})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())