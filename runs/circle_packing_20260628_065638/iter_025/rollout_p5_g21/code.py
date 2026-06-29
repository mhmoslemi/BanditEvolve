import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized initialization with asymmetric grid clustering, staggered layout, and perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset with asymmetric range
        x = x_center + np.random.uniform(-0.07, 0.04)  # Asymmetric horizontal perturbation
        y = y_center + np.random.uniform(-0.05, 0.06)  # Asymmetric vertical perturbation
        # Staggered grid for alternate rows
        if row % 2 == 1:
            x += 0.5 / cols
        # Add small random noise for escape from symmetry
        xs.append(x + np.random.uniform(-0.005, 0.005))
        ys.append(y + np.random.uniform(-0.005, 0.005))
    
    # Optimized initial radii based on grid and perturbation scaling
    r0 = 0.36 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Consistent length for 3*26=78

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective is to maximize sum_radii

    # Vectorized constraint functions with stable lambda closures
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Optimized pairwise overlap constraints using vectorized function
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                          "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                          - (v[3*i+2] + v[3*j+2])**2})

    # Asymmetric reconfiguration with stochastic spatial perturbation
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})

    if res.success:
        v = res.x
        # Calculate and identify the least constrained circle: circle with largest min distance to neighbors
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        min_dist = np.min(dists, axis=1)
        isolated_idx = np.argmax(min_dist)  # Circle with largest minimal distance

        # Create asymmetric geometric hash for reconfiguration
        random_hash = np.random.rand(n, 2) * 0.04
        # Add asymmetric spatial perturbation
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0] * 0.9
            perturbed_v[3*i+1] += random_hash[i, 1] * 0.95

        # Re-evaluate with asymmetric perturbation and reconfig
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Final optimization with targeted radius expansion on most isolated circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Calculate distances and find most isolated circle
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
        min_dist = np.min(dists, axis=1)
        isolated_idx = np.argmax(min_dist)

        # Calculate expansion factor to slightly increase isolated circle's radius
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.006  # Small but consistent expansion
        expansion_factor = (target_total_sum - total_sum) / (n - 1)

        # Apply expansion with soft boundaries
        v_new = v.copy()
        v_new[2::3] = radii.copy()
        # Expand only the isolated circle
        v_new[3*isolated_idx+2] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != isolated_idx:
                v_new[3*i+2] += expansion_factor * (1.0 + 0.1 * np.random.rand())  # Stochastic growth

        # Re-optimize with expanded radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())