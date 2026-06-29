import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with hybrid staggered and radial clustering for diverse distribution
    xs = []
    ys = []
    for i in range(n):
        if i % 2 == 0:
            # Regular grid with staggered rows
            row = i // cols
            col = i % cols
            x_center = (col + 0.5) / cols
            y_center = (row + 0.5) / rows
            x = x_center + np.random.uniform(-0.05, 0.05)
            y = y_center + np.random.uniform(-0.05, 0.05)
            if row % 2 == 1:
                x += 0.5 / cols
        else:
            # Radial distribution for diverse placement
            angle = 2 * np.pi * i / n
            radius = 0.6
            x = radius * np.cos(angle) + 0.5
            y = radius * np.sin(angle) + 0.5
            x += np.random.uniform(-0.03, 0.03)
            y += np.random.uniform(-0.03, 0.03)
        xs.append(x)
        ys.append(y)
    
    r0 = (0.5 / cols) * (n / 24) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n-length match

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints using closure-based parameter capturing
    cons = []
    for i in range(n):
        # Left edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top edge constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized pairwise radius constraint with optimized lambda closure captures
    for i in range(n):
        for j in range(i + 1, n):
            # Use functools.lru_cache with limited size to maintain optimization efficiency
            # This avoids closure capture issues with non-hashable variables
            # Note: In Python 3.10+, functools.lru_cache can be used on lambda functions with
            # args that are hashable (like integers). For performance, we'll precompute indices

            def make_constraint(i, j):
                def constraint_func(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                return constraint_func
            
            cons.append({"type": "ineq", "fun": make_constraint(i, j)})

    # Initial optimization with increased precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Asymmetric spatial stochastic reconfiguration with adaptive spatial hashing
    if res.success:
        v = res.x
        radii_base = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Spatial hashing matrix for refined perturbation
        spatial_hash = np.random.rand(n, 2) * 0.05
        
        # Apply spatial reconfiguration with radius sensitivity
        perturbed_v = v.copy()
        for i in range(n):
            # Use radius to scale perturbation size for constrained circles
            scale = 1.0 if radii_base[i] < np.mean(radii_base) else 1.2
            perturbed_v[3*i] += spatial_hash[i, 0] * scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale
        
        # Re-optimization with perturbations
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted gradient-free radius expansion heuristic
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Calculate pairwise distances using vectorized broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle by maximizing minimum distances
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate optimal growth increment based on distribution and current state
        current_total = np.sum(radii)
        avg_radius = np.mean(radii)
        max_potential_growth = 0.006 * (1.0 / (np.sum((dists > 0) * 1.0) * 1.0))

        # Apply exponential growth strategy to least constrained circle
        # with controlled spatial expansion
        expansion_factor = (max_potential_growth / 2.0) * (radii[least_constrained_idx] / avg_radius)
        for i in range(n):
            if i == least_constrained_idx:
                # Apply exponential expansion with constraint-aware limit
                growth = 1.1 * expansion_factor * (1.0 + 0.3 * np.random.rand())
                v[3*i+2] += growth
            else:
                # Slight spatial coupling to maintain distribution balance
                v[3*i+2] += expansion_factor * (1.0 + 0.2 * np.random.rand())
        
        # Re-evaluate with expanded configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Final cleanup and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())