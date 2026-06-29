import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized but structured grid and enhanced symmetry avoidance
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x_center = (col + 0.5) / cols * 1.05  # Slight stretch to reduce boundary pressure
        y_center = (row + 0.5) / rows * 1.05
        # Randomized offset to avoid clustering, with adaptive spread
        offset_x = np.random.uniform(-0.04, 0.04)
        offset_y = np.random.uniform(-0.04, 0.04)
        # Alternate row offset for staggered layout
        row_offset = 0.0 if row % 2 == 0 else 0.5 / cols
        x = x_center + offset_x
        y = y_center + offset_y + row_offset
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure bounds list has 3*n entries for the vector of length 3n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left boundary constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            # Efficient computation of distance squared minus sum of radii squared
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    # Jiggle heuristic: perturb the smallest circles to escape local minima
    if res.success:
        v = res.x
        radii_val = v[2::3]
        # Identify smallest circles to perturb
        min_radius_mask = radii_val < np.mean(radii_val) - 0.003
        indices_to_perturb = np.where(min_radius_mask)[0]
        # Generate perturbations based on spatial context to avoid boundary collisions
        perturbation_factor = 0.025
        perturbation_strength = np.zeros(3 * n)
        for idx in indices_to_perturb:
            # Calculate current position and radius
            x, y, r = v[3*idx], v[3*idx+1], v[3*idx+2]
            # Add directional perturbations to neighboring regions
            dx = (np.random.uniform(-0.05, 0.05) * r) / np.sqrt(2)  # Add radial perturbation
            dy = (np.random.uniform(-0.05, 0.05) * r) / np.sqrt(2)  # Add radial perturbation
            perturbation_strength[3*idx] = dx
            perturbation_strength[3*idx+1] = dy
            # Apply perturbation
            v[3*idx] += dx
            v[3*idx+1] += dy
            # Re-evaluate with new perturbed parameters
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-12})

    # Final optimization with increased precision
    if res.success:
        v = res.x
        # Add a final small expansion pass to target under-constrained circles
        radii_val = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Compute pairwise distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify circles with greatest allowable expansion
        min_dists = np.min(dists, axis=1)
        expansion_factor = 0.006  # Targeted expansion amount
        expansion_mask = min_dists > 0.7 * (radii_val + np.min(radii_val))  # Expand circles with more space
        
        # Create new radii with expansion
        new_radii = radii_val.copy()
        for i in range(n):
            if expansion_mask[i]:
                # Calculate how much we could expand without violating overlap
                max_expansion = 0.0
                for j in range(n):
                    if j != i and dists[i,j] < (radii_val[i] + radii_val[j] - 1e-8):
                        # Could expand as long as it doesn't cause overlap
                        # Assuming perfect expansion, the new distance is (r_i + r_j)
                        new_dist = (radii_val[i] + new_radii[i]) + (radii_val[j] + new_radii[j])
                        available_expansion = (new_dist - dists[i,j]) / 2
                        max_expansion = max(max_expansion, available_expansion)
                # Expand by a fraction of the target amount
                if max_expansion > 0:
                    new_radii[i] += expansion_factor * (max_expansion / np.max([max_expansion, 1e-6]))  # Safety cap
                    
        # Apply expansion and re-evaluate
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())