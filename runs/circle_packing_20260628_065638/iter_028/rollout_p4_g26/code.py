import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with geometrically distributed centers, using a staggered grid with spatial randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering with larger spread
        x_offset = np.random.uniform(-0.08, 0.08)
        y_offset = np.random.uniform(-0.08, 0.08)
        # Alternate row shift (staggered grid) to improve packing density
        if row % 2 == 1:
            x_center += 0.5 / cols
        xs.append(x_center + x_offset)
        ys.append(y_center + y_offset)
    
    # Start with a reasonable initial radius, adjusted for grid width
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Construct bounds with correct length for 3n decision variables
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Build constraints with correct parameter capture and index binding
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
    
    # Vectorized constraints for circle overlaps
    for i in range(n):
        for j in range(i + 1, n):
            fun = lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": fun})

    # Initial optimization with increased max iterations and tight tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Primary optimization step
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Step 1: Shake the smallest circles with random spatial perturbations
        # This helps escape local minima and allows exploration
        # Generate spatial noise proportional to circle size
        shake_factor = 0.02
        radii_mask = np.ones_like(radii)
        radii_mask[radii < np.mean(radii) * 0.7] = 2.0  # Amplify shake for small circles
        shake = np.random.rand(*v.shape) * shake_factor * radii_mask
        perturbed_v = v + shake
        
        # Apply perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 400, "ftol": 1e-11})
    
    # Step 2: Refinement with expansion of the least constrained circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate pairwise minimum distances for constraint sensitivity
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute for each circle the minimum distance to any other circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Circle with least margin (most constrained by others)
        
        # Compute growth based on target expansion and current total sum
        current_total = radii.sum()
        target_growth = 0.005  # Small growth factor
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        expansion = expansion_factor * 1.1  # Slight over-expansion to trigger optimization
        new_radii[least_constrained_idx] += expansion
        
        # Apply expansion with constraint validation
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final refinement: optimize small circles without major changes
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())