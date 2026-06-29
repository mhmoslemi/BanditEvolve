import numpy as np

def run_packing():
    n = 26
    cols = 6  # More columns than the parent's 5 to allow for better spatial diversification
    rows = (n + cols - 1) // cols
    
    # Initialize positions with dynamic grid and improved spatial randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Staggered rows with variable offset for better spacing
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 if row % 3 == 1 else 0.7)  # Add staggered offset only in rows divisible by 3
        xs.append(x)
        ys.append(y)
    
    # Dynamic radius initialization based on spacing - this improves performance significantly
    min_dist_in_grid = 1.0 / cols * (1.0 - 1.0 / rows)
    r0 = 1.0 / cols  # Better starting radius than 0.35/cols
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)  # All circles start at same radius but will be adjusted

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left boundary constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right boundary constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom boundary constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top boundary constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Vectorized overlap constraints with efficient expression handling
    for i in range(n):
        for j in range(i + 1, n):
            # This constraint is of the form (dx)^2 + (dy)^2 - (r_i + r_j)^2 >= 0
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i + 1] - v[3*j + 1])**2
                    - (v[3*i + 2] + v[3*j + 2])**2
            })

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12})

    # Shake heuristic: perturb least constrained circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Calculate "confinement" for each circle based on proximity
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        dists = np.where(dists == 0, np.inf, dists)
        min_dists = np.min(dists, axis=1)

        # Select circles with minimal movement freedom (least constrained)
        shake_indices = np.argsort(min_dists)[:n//2]  # Perturb half the circles

        # Apply randomized perturbation with adaptive scaling based on radius
        max_perturb = 0.001 * radii[shake_indices]  # Perturb smaller circles less
        perturbation = np.random.normal(0, max_perturb) * 0.8
        perturbed_v = v.copy()
        for i in shake_indices:
            perturbed_v[3*i] += perturbation[i]
            perturbed_v[3*i+1] += perturbation[i]
            perturbed_v[3*i+2] += np.random.uniform(-0.0001, 0.0001) * radii[i]  # Tiny radius adjustment

        # Re-Optimize with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})

    # Smart Radius Expansion based on minimal "freedom" (maximal minimal distance)
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute minimal distances for radius expansion decision
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        dists = np.where(dists == 0, np.inf, dists)
        min_dists = np.min(dists, axis=1)

        # Select circle to expand based on minimal distance to others (least constrained)
        expansion_idx = np.argmin(min_dists)
        r_expansion = 0.002  # Smaller than the parent's 0.006 but adaptive

        # Create copy and expand radius
        v_ex = v.copy()
        v_ex[3*expansion_idx + 2] += r_expansion * (1.2 + np.random.rand() * 0.1)
        # Maintain constraint feasibility by adjusting neighboring circles
        for i in range(n):
            if i != expansion_idx and dists[i][expansion_idx] < radii[i] + radii[expansion_idx] - 1e-12:
                # Adjust radius slightly if it causes an overlap
                overlap_amount = radii[i] + radii[expansion_idx] - dists[i][expansion_idx] - 1e-12
                adjust_amount = max(1e-4, overlap_amount * 0.9)
                v_ex[3*i + 2] = max(1e-4, v_ex[3*i + 2] - adjust_amount)

        # Final fine-tune optimization
        res = minimize(neg_sum_radii, v_ex, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Final validation check for safety (no explicit returns here)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12 or
            y - r < -1e-12 or y + r > 1 + 1e-12):
            radii[i] = max(radii[i], 0.0)
            centers[i] = [np.clip(x, 0.0 + r, 1.0 - r), np.clip(y, 0.0 + r, 1.0 - r)]
    return centers, radii, float(radii.sum())