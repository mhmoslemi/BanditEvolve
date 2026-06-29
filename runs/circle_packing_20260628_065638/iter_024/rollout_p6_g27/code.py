import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with staggered grid + randomized offsets
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add random offset to break symmetry
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Stagger rows to reduce clustering
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
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n entries for x, y, r for 26 circles

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Minimize the negative sum to maximize radii

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # x must be >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # x must be <= 1 - r
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y must be >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # y must be <= 1 - r
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-10})

    if res.success:
        v = res.x
        # Apply randomized geometric hashing for reconfiguration
        hash_perturb = np.random.rand(n, 2) * 0.06  # small random displacement
        perturbed_v = v.copy()
        perturbed_v[0::3] += hash_perturb[:, 0]
        perturbed_v[1::3] += hash_perturb[:, 1]

        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Find least constrained circle (minimize minimum distance to others)
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)

        min_distances = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_distances)

        # Controlled radius expansion to the least constrained circle
        expansion_factor = 0.006 / (n - 1)  # Small controlled expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor

        # Update decision vector with expanded radii and re-optimize
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    # Final solution
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())