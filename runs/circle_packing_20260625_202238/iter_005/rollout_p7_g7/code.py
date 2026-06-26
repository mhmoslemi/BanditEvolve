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

    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Compute constraint tightness
    constraint_tightness = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                constraint_tightness[i] += (v[3*i+2] + v[3*j+2] - dist)
                constraint_tightness[j] += (v[3*i+2] + v[3*j+2] - dist)
    
    # Sort indices by constraint tightness (most constrained first)
    sorted_indices = np.argsort(constraint_tightness)
    
    # Permute the decision vector based on sorted indices
    permuted_v = np.zeros_like(v)
    for i, idx in enumerate(sorted_indices):
        permuted_v[3*i] = v[3*idx]
        permuted_v[3*i+1] = v[3*idx+1]
        permuted_v[3*i+2] = v[3*idx+2]
    
    # Re-optimize with permuted initial guess
    res = minimize(neg_sum_radii, permuted_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else permuted_v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Identify most restrictive constraint
    max_tightness = np.max(constraint_tightness)
    if max_tightness > 1e-6:
        i_most_restrictive = np.argwhere(constraint_tightness == max_tightness)[0][0]
        # Remove the most restrictive constraint temporarily
        cons = [c for c in cons if not (c["fun"].__name__ == "constraint_func" and 
                                        c["fun"].__code__.co_freevars[0] == i_most_restrictive)]
        # Add a modified objective function that prioritizes radius expansion
        def modified_neg_sum_radii(v):
            # Calculate current sum of radii
            current_sum = np.sum(v[2::3])
            # Calculate the potential expansion of the most restrictive circle
            i = i_most_restrictive
            radius = v[3*i+2]
            # Calculate the maximum possible expansion without violating other constraints
            max_expansion = 0
            for j in range(n):
                if j == i:
                    continue
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radius + v[3*j+2] - 1e-5:
                    max_expansion = max(max_expansion, (radius + v[3*j+2] - dist))
            # Add penalty for violating the removed constraint
            penalty = max(0, (radius + v[3*i+2] - 1e-5 - max_expansion))
            return -current_sum + 0.1 * penalty

        # Re-optimize with modified objective
        res = minimize(modified_neg_sum_radii, permuted_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
        v = res.x if res.success else permuted_v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
        # Re-add the most restrictive constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i_most_restrictive: 
                     (v[3*i] - v[3*i+2])**2 + (v[3*i+1] - v[3*i+2])**2 - (v[3*i+2] + v[3*i+2])**2})
    
    return centers, radii, float(radii.sum())