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

    # Add a topological reconfiguration heuristic
    def topological_reconfiguration(v):
        # Group circles into independent components (e.g., rows)
        component_indices = np.arange(n)
        # Apply a global permutation of the component indices
        np.random.shuffle(component_indices)
        # Apply the permutation to the decision vector
        permuted_v = np.zeros_like(v)
        for i in range(n):
            permuted_v[3*i] = v[3*component_indices[i]]
            permuted_v[3*i+1] = v[3*component_indices[i]+1]
            permuted_v[3*i+2] = v[3*component_indices[i]+2]
        return permuted_v

    # Add a constraint that forces at least one component to expand its radii
    def force_expansion(v):
        # Group circles into independent components (e.g., rows)
        component_indices = np.arange(n)
        # Identify the largest component
        component_sizes = np.bincount(component_indices)
        largest_component = np.argmax(component_sizes)
        # Force expansion of the largest component
        v_expanded = np.copy(v)
        for i in range(n):
            if component_indices[i] == largest_component:
                v_expanded[3*i+2] = np.clip(v_expanded[3*i+2] * 1.1, 1e-4, 0.5)
        return v_expanded

    # Apply the topological reconfiguration and expansion heuristic
    v_reconfigured = topological_reconfiguration(v0)
    v_expanded = force_expansion(v_reconfigured)
    res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())