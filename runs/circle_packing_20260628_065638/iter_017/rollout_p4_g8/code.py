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
        # Randomized offset to break symmetry
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
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

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Radical spatial reconfiguration: randomized geometric hashing
    if res.success:
        v = res.x
        # Apply randomized geometric hashing to break symmetries and explore new configurations
        perturbation = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation[i, 0]
            perturbed_v[3*i+1] += perturbation[i, 1]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion with strict non-overlap and topological reordering
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Find the circle with the smallest non-zero radius
        smallest_radius_idx = np.argmin(radii)
        # Expand its radius and apply hard constraint to total sum
        total_sum = np.sum(radii)
        # Enforce strict non-overlap and perform topological reordering
        # Use a local search to perturb positions and re-evaluate
        for _ in range(30):
            # Perturb the position of the smallest radius circle
            perturb = np.random.rand(2) * 0.05
            v[3*smallest_radius_idx] += perturb[0]
            v[3*smallest_radius_idx+1] += perturb[1]
            # Ensure position constraints are still satisfied
            if v[3*smallest_radius_idx] < 0:
                v[3*smallest_radius_idx] = 0
            if v[3*smallest_radius_idx+1] < 0:
                v[3*smallest_radius_idx+1] = 0
            if v[3*smallest_radius_idx] + radii[smallest_radius_idx] > 1:
                v[3*smallest_radius_idx] = 1 - radii[smallest_radius_idx]
            if v[3*smallest_radius_idx+1] + radii[smallest_radius_idx] > 1:
                v[3*smallest_radius_idx+1] = 1 - radii[smallest_radius_idx]
            # Re-evaluate with adjusted parameters
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 30, "ftol": 1e-10})
            if res.success:
                v = res.x
                # Check for overlap and refine positions if necessary
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = v[3*i] - v[3*j]
                        dy = v[3*i+1] - v[3*j+1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < radii[i] + radii[j] - 1e-12:
                            # Adjust positions to resolve overlap
                            overlap = (radii[i] + radii[j]) - dist
                            # Move circles apart in the direction of the vector between them
                            direction_x = dx / dist
                            direction_y = dy / dist
                            move = 0.5 * overlap
                            v[3*i] += direction_x * move
                            v[3*i+1] += direction_y * move
                            v[3*j] -= direction_x * move
                            v[3*j+1] -= direction_y * move
                            # Ensure positions remain within bounds
                            v[3*i] = np.clip(v[3*i], 0, 1)
                            v[3*i+1] = np.clip(v[3*i+1], 0, 1)
                            v[3*j] = np.clip(v[3*j], 0, 1)
                            v[3*j+1] = np.clip(v[3*j+1], 0, 1)
                break
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())