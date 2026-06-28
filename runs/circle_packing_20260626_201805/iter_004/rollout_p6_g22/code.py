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

    # Topological reconfiguration: split circles into components and permute
    def create_components(v):
        # Split into 2 components based on their initial positions
        component1 = v[0::3][0:13]
        component1_y = v[1::3][0:13]
        component1_r = v[2::3][0:13]
        
        component2 = v[0::3][13:]
        component2_y = v[1::3][13:]
        component2_r = v[2::3][13:]
        
        # Permute the components
        permuted_component1 = np.concatenate([component2, component1])
        permuted_component1_y = np.concatenate([component2_y, component1_y])
        permuted_component1_r = np.concatenate([component2_r, component1_r])
        
        # Reconstruct the vector
        new_v = np.zeros_like(v)
        new_v[0::3] = permuted_component1
        new_v[1::3] = permuted_component1_y
        new_v[2::3] = permuted_component1_r
        return new_v

    # Add a constraint that at least one component expands its radii without overlapping
    def component_expansion_constr(v):
        # Split into 2 components
        component1 = v[0::3][0:13]
        component1_y = v[1::3][0:13]
        component1_r = v[2::3][0:13]
        
        component2 = v[0::3][13:]
        component2_y = v[1::3][13:]
        component2_r = v[2::3][13:]
        
        # Check if expansion is possible for component 1
        min_distance = np.inf
        for i in range(13):
            for j in range(i + 1, 13):
                dx = component1[i] - component1[j]
                dy = component1_y[i] - component1_y[j]
                dist = np.sqrt(dx**2 + dy**2)
                min_distance = min(min_distance, dist - (component1_r[i] + component1_r[j]))
        
        # Check if expansion is possible for component 2
        for i in range(13):
            for j in range(i + 1, 13):
                dx = component2[i] - component2[j]
                dy = component2_y[i] - component2_y[j]
                dist = np.sqrt(dx**2 + dy**2)
                min_distance = min(min_distance, dist - (component2_r[i] + component2_r[j]))
        
        return min_distance

    cons.append({"type": "ineq", "fun": component_expansion_constr})

    # Apply the topological reconfiguration to the initial guess
    v_reconfigured = create_components(v0)
    res = minimize(neg_sum_radii, v_reconfigured, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())