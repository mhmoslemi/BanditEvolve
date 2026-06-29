import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    def initial_layout():
        xs = []
        ys = []
        # Staggered grid with randomized offsets and asymmetric perturbation
        for i in range(n):
            row = i // cols
            col = i % cols
            x_center = (col + 0.5) / cols
            y_center = (row + 0.5) / rows
            # Initial offset for diversity
            x = x_center + np.random.uniform(-0.1, 0.1)
            y = y_center + np.random.uniform(-0.1, 0.1)
            # Stagger alternate rows
            if row % 2 == 1:
                x += 0.4 / cols
            xs.append(x)
            ys.append(y)
        return xs, ys

    # Initialize with asymmetric grid perturbation
    xs, ys = initial_layout()
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # matches v length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Construct constraints with fixed indices
    cons = []
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Create overlap constraints using vectorized distance calculation
    for i in range(n):
        for j in range(i + 1, n):
            # Precompute the index offsets for i-th and j-th circle
            i_offset = 3 * i
            j_offset = 3 * j
            def constraint_func(v, i=i, j=j, i_offset=i_offset, j_offset=j_offset):
                dx = v[i_offset] - v[j_offset]
                dy = v[i_offset+1] - v[j_offset+1]
                return dx*dx + dy*dy - (v[i_offset+2] + v[j_offset+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization pass
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-11})

    # Disruptive transformation phase: random geometric hashing
    if res.success:
        v = res.x
        # Compute distance matrix with vectorized broadcasting
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        radii = v[2::3]
        adjacency_mask = dists <= (radii + radii.reshape(-1, 1))
        from scipy.sparse import csr_matrix, csgraph
        graph = csr_matrix(adjacency_mask)
        components = csgraph.connected_components(graph)[1]

        # Apply asymmetric stochastic hashing for layout reordering
        component_hash = np.random.rand(n, 2) * 0.06
        # Apply spatial hashing to move circles between components
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += component_hash[components[i], 0] * (1 + np.random.rand())
            perturbed_v[3*i+1] += component_hash[components[i], 1] * (1 + np.random.rand())
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11})

    # Radius expansion with forced topological reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2
                       + (centers[:, np.newaxis, 1] - centers[np.newaxis, 1])**2)
        
        # Find the circle with the smallest radius
        smallest_radius_idx = np.argmin(radii)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Force expansion of least constrained circle while maintaining constraints
        target_total_sum = np.sum(radii) + 0.008  # More aggressive expansion
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)

        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Over-expand to trigger reconfiguration
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())  # Stochastic expansion

        # Validate expansion
        while True:
            expanded_centers = np.column_stack([v[0::3], v[1::3]])
            expanded_radii = new_radii
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < expanded_radii[i] + expanded_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # Reduce expansion if validation fails
                new_radii = radii + (new_radii - radii) * 0.98

        # Apply expansion to decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())