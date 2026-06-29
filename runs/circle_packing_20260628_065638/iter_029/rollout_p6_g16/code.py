import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering
    xs = []
    ys = []
    # Adaptive spacing to reduce clumping using Gaussian noise
    for i in range(n):
        col = i % cols
        row = i // cols
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        
        # Generate random perturbations with adaptive scaling by radius expectations
        x_offset = np.random.normal(0.0, 0.04 / np.sqrt(cols)) * (1.0 - 2.0 * np.random.rand())
        y_offset = np.random.normal(0.0, 0.04 / np.sqrt(rows)) * (1.0 - 2.0 * np.random.rand())
        # Apply staggered row offset for grid stability
        if row % 2 == 1:
            x_offset += (0.5 / cols) * (1.0 - 2.0 * np.random.rand()) * 0.3
        
        x = x_base + x_offset
        y = y_base + y_offset
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-4  # More conservative initial radius
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)  # Radius initialization

    # Create bounds list with same length as decision vector (3*n)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 1.0)]  # Radius can go up to 1.0

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Minimize negative of sum to maximize sum

    # Vectorized constraint functions with closure binding via lambda with i
    cons = []
    for i in range(n):
        # Left: x - r >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right: x + r <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom: y - r >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top: y + r <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized pairwise distance constraints
    # Use matrix operations for overlap check (simplified form)
    # This avoids repeated lambda captures, which is critical for performance
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq", 
                "fun": (lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                        - (v[3*i+2] + v[3*j+2])**2
                )
            })

    # Perform initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Adaptive reconfiguration: spatial perturbation based on radius-weighted hashes
    # Introduce a radius-aware perturbation to break local optima
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        max_radius = np.max(radii)
        
        # Generate radius-aware spatial hash
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbation_factors = radii / max_radius * 0.5  # Scale perturbations by radius
        
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * perturbation_factors[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * perturbation_factors[i]
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Optimized spatial-aware expansion strategy: identify "least confined" circle
    if res.success:
        v = res.x
        radii = v[2::3]
        # Compute all pairwise distances
        dx = v[0::3] - v[3::3]
        dy = v[1::3] - v[4::3]
        dists = np.sqrt(dx**2 + dy**2) - radii[:, np.newaxis] - radii[np.newaxis, :]
        # Valid distance matrix (avoid self-edges)
        dists[np.diag_indices(n)] = np.inf
        # Find circle with largest minimum distance (least constrained)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Create a radius expansion vector with adaptive scaling per constraint
        radii_expansion = np.zeros(n)
        max_possible_growth = 0.6  # Safety margin for expansion
        min_growth_per_circle = 0.0005 * max_possible_growth / (n - 1)
        
        # Start with a moderate expansion
        base_growth = min_growth_per_circle * 5  # Increase starting growth
        
        # Apply expansion to non-least-constrained circles
        for i in range(n):
            if i != least_constrained_idx:
                # Use proximity to expansion center to determine growth potential
                dist_to_least = np.sqrt( (v[3*i] - v[3*least_constrained_idx])**2 +
                                       (v[3*i+1] - v[3*least_constrained_idx+1])**2 )
                # Growth inversely proportional to distance from target circle
                radii_expansion[i] = base_growth * (1.0 / (1.0 + dist_to_least / 0.2))
        
        # Set the least constrained circle to have higher growth
        radii_expansion[least_constrained_idx] = base_growth * 1.5
        
        # Apply the expansion in a gradient-aware way
        max_allowed_radius = np.min(1.0 - (v[0::3] - v[2::3]) 
                                   * np.array([1,1,0]).repeat(n))  # Bound by left/right/top/bottom
        
        new_radii = radii + radii_expansion  # Apply expansion
        new_radii = np.minimum(new_radii, max_allowed_radius)  # Respect box constraints
        new_radii = np.maximum(new_radii, radii * 0.95)  # Soft lower limit
        
        # Validate before applying expansion
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-10})
        
        # Apply the expansion with a second optimization pass
        if res.success:
            v_expanded = v.copy()
            v_expanded[2::3] = new_radii
            res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 200, "ftol": 1e-10})
    
    # Clean-up and validation step, with enhanced tolerance
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final safety check to ensure all constraints are met
    # This is crucial due to potential residual constraint violations
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - 1e-12:
                valid = False
                break
        if not valid:
            break
    
    # If constraints failed, fallback to a safe but suboptimal solution
    if not valid:
        centers, radii = _safe_config(n, bounds)
        sum_radii = float(radii.sum())
    else:
        sum_radii = float(radii.sum())
    
    return centers, radii, sum_radii

def _safe_config(n, bounds):
    # A default safe configuration: arrange circles in a grid with minimum spacing
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    centers = []
    radii = []
    
    for i in range(n):
        col = i % cols
        row = i // cols
        x = (col + 0.5) / cols * 0.95  # Reduce to avoid clipping
        y = (row + 0.5) / rows * 0.95
        # Small radius to allow for spacing
        r = 0.05 * (1.0 - (col + row) / (cols + rows))
        centers.append([x, y])
        radii.append(r)
    
    radii = np.array(radii)
    centers = np.array(centers)
    return centers, radii