import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Randomized offset to break symmetry
        x += np.random.uniform(-0.05, 0.05)
        y += np.random.uniform(-0.05, 0.05)
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
    
    # Trigger topological overhaul with randomized geometric hashing
    if res.success:
        v = res.x
        # Generate new random positions with geometric hashing
        new_positions = np.random.rand(n, 2)
        new_positions[:, 0] = np.sort(new_positions[:, 0])
        new_positions[:, 1] = np.sort(new_positions[:, 1])
        # Normalize to unit square
        new_positions /= np.max(new_positions, axis=0)
        # Compute initial radii based on distance to neighbors
        dists = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dists[i] += np.sqrt((new_positions[i, 0] - new_positions[j, 0])**2 + (new_positions[i, 1] - new_positions[j, 1])**2)
        avg_dist = np.mean(dists)
        r0 = avg_dist / (2 * np.sqrt(n)) - 1e-3
        v_new = np.empty(3 * n)
        v_new[0::3] = new_positions[:, 0]
        v_new[1::3] = new_positions[:, 1]
        v_new[2::3] = np.full(n, r0)
        # Re-evaluate with new parameters
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Radical radius expansion for smallest circle
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Identify smallest circle
        smallest_idx = np.argmin(radii)
        # Expand its radius while maintaining non-overlap
        v[3*smallest_idx + 2] += 0.005
        # Perturb its position slightly to allow expansion
        v[3*smallest_idx] += np.random.uniform(-0.01, 0.01)
        v[3*smallest_idx+1] += np.random.uniform(-0.01, 0.01)
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())