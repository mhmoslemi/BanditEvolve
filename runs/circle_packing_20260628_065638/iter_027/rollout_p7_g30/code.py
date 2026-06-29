import numpy as np

def run_packing():
    n = 26
    
    # Adaptive grid and initial configuration with geometric awareness
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Use a more informed initial spatial layout
    xs = []
    ys = []
    
    # First, create a grid with rows and columns
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add jitter to break symmetry but preserve geometric sense
        x_jitter = np.random.uniform(-0.04, 0.04)
        y_jitter = np.random.uniform(-0.04, 0.04)
        
        # Shift alternate rows to form staggered pattern
        if row % 2 == 1:
            x_center += 0.5 / cols
        # Adjust to keep circles inside the square with buffer
        x_center = np.clip(x_center + x_jitter, 0.05, 0.95)
        y_center = np.clip(y_center + y_jitter, 0.05, 0.95)
        
        xs.append(x_center)
        ys.append(y_center)
    
    # Initial radii
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
    
    # Vectorized constraints for boundaries with proper lambda binding
    cons = []
    for i in range(n):
        # Left: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right: x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top: y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with lambda closures
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})
    
    # Initial optimization with aggressive settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # First shake: reconfiguration through spatial perturbations
    if res.success:
        v = res.x
        # Spatial reconfiguration with adaptive perturbations based on circle sizes
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        radii = v[2::3]
        scale_factor = (np.max(radii) / np.min(radii)) * 0.1
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * scale_factor
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * scale_factor
        # Re-evaluate
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Second shake: reconfiguration of smallest circles to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        # Get the smallest circle
        min_radius_idx = np.argmin(radii)
        # Perturb its position and re-optimize with smaller step
        perturb = np.random.rand(2) * 0.1
        v[3*min_radius_idx] += perturb[0]
        v[3*min_radius_idx+1] += perturb[1]
        # Re-evaluate
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Third shake: adaptive radius increase for non-overlapping circles
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Compute pairwise distances and non-overlap constraints
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate minimum distances for each circle to others
        min_dists = np.min(dists, axis=1)
        # Identify circles that could be expanded with most room
        expandable_indices = np.argsort(min_dists)[::-1]
        
        # Create a mask to ensure we don't expand overlapping circles
        expand_mask = np.ones(n, dtype=bool)
        # First expand the most expandable circle that is not overlapping
        for idx in expandable_indices:
            # Check if circle idx can be expanded without overlap
            if not expand_mask[idx]:
                continue
            for j in range(n):
                if j == idx:
                    continue
                if not expand_mask[j]:
                    continue
                if dists[idx, j] < radii[idx] + radii[j] - 1e-10:
                    expand_mask[idx] = False
                    break
        
        # Calculate total current sum
        total_current = np.sum(radii)
        # Targeted radius expansion
        expansion_amount = (0.008) / (n - 1)  # 0.008 extra total
        for idx in expandable_indices:
            if expand_mask[idx]:
                v[3*idx + 2] += expansion_amount
        
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())