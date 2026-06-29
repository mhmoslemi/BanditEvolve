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
    
    # Vectorized overlap constraints with geometric hashing
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
    
    # Asymmetric reconfiguration: randomize spatial constraints with geometric hashing
    if res.success:
        v = res.x
        # Introduce geometric hashing for spatial reconfiguration
        hash_map = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_map[i, 0]
            perturbed_v[3*i+1] += hash_map[i, 1]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion on least constrained circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Find the circle with the smallest non-zero radius
        smallest_radius_idx = np.argmin(radii)
        # Expand its radius while ensuring non-overlap
        # First, ensure the smallest radius is below the average
        avg_radius = np.mean(radii)
        if radii[smallest_radius_idx] < avg_radius - 1e-3:
            # Expand it while maintaining feasible non-overlap
            for _ in range(10):
                # Try expanding the smallest radius by a small amount
                expansion = 0.0005
                new_radii = radii.copy()
                new_radii[smallest_radius_idx] += expansion
                # Check if expansion is feasible
                feasible = True
                for i in range(n):
                    for j in range(i+1, n):
                        dx = centers[i, 0] - centers[j, 0]
                        dy = centers[i, 1] - centers[j, 1]
                        if np.sqrt(dx*dx + dy*dy) < new_radii[i] + new_radii[j] - 1e-12:
                            feasible = False
                            break
                    if not feasible:
                        break
                if feasible:
                    radii = new_radii
                else:
                    # If not feasible, adjust the positions of the smallest circle
                    # Perturb its position slightly and re-optimize
                    perturbation = np.random.uniform(-0.01, 0.01, size=2)
                    perturbed_centers = centers.copy()
                    perturbed_centers[smallest_radius_idx] += perturbation
                    perturbed_v = np.zeros(3*n)
                    perturbed_v[0::3] = perturbed_centers[:, 0]
                    perturbed_v[1::3] = perturbed_centers[:, 1]
                    perturbed_v[2::3] = radii.copy()
                    res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                                   constraints=cons, options={"maxiter": 100, "ftol": 1e-10})
                    if res.success:
                        v = res.x
                        centers = np.column_stack([v[0::3], v[1::3]])
                        radii = v[2::3]
                    break
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())