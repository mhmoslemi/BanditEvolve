import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorize the overlap constraints for better performance
    def vectorized_overlap_constraint(v):
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        dx = x_centers[:, np.newaxis] - x_centers[np.newaxis, :]
        dy = y_centers[:, np.newaxis] - y_centers[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r_radii[:, np.newaxis] + r_radii[np.newaxis, :])**2
        return dist_sq - min_dist_sq

    # Convert to list of functions for each pair
    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            overlap_cons.append({"type": "ineq", "fun": constraint_func})

    cons.extend(overlap_cons)

    # Introduce a radical topological reconfiguration
    def topological_reconfiguration(v):
        # Group circles into independent components
        component_sizes = [5, 5, 5, 5, 6]  # Distribute circles into 5 components
        component_indices = []
        start_idx = 0
        for size in component_sizes:
            component_indices.append(list(range(start_idx, start_idx + size)))
            start_idx += size
        
        # Permute the components
        np.random.shuffle(component_indices)
        new_v = np.copy(v)
        
        # Reorder the decision vector
        new_order = []
        for comp in component_indices:
            new_order.extend(comp)
        
        # Apply the permutation
        permuted_v = np.zeros_like(v)
        for i, idx in enumerate(new_order):
            permuted_v[3*i] = v[3*idx]
            permuted_v[3*i+1] = v[3*idx+1]
            permuted_v[3*i+2] = v[3*idx+2]
        
        # Add a constraint to force at least one component to expand
        def expansion_constraint(v):
            # Check if any component has a minimum distance between circles
            x_centers = v[0::3]
            y_centers = v[1::3]
            r_radii = v[2::3]
            min_dist = np.inf
            for comp in component_indices:
                comp_x = x_centers[comp]
                comp_y = y_centers[comp]
                comp_r = r_radii[comp]
                dists = np.zeros(len(comp))
                for i in range(len(comp)):
                    for j in range(i + 1, len(comp)):
                        dx = comp_x[i] - comp_x[j]
                        dy = comp_y[i] - comp_y[j]
                        dist = np.sqrt(dx**2 + dy**2)
                        dists[i] = np.min([dists[i], dist])
                        dists[j] = np.min([dists[j], dist])
                min_dist = np.min([min_dist, np.min(dists)])
            return min_dist - (np.sum(r_radii[component_indices[0]]) / len(component_indices[0]))  # Force at least one component to expand

        # Add the expansion constraint
        cons.append({"type": "ineq", "fun": expansion_constraint})
        return permuted_v

    # Apply the topological reconfiguration to the initial guess
    v_reconfigured = topological_reconfiguration(v0)
    res = minimize(neg_sum_radii, v_reconfigured, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())