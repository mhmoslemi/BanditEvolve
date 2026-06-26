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
    
    # Precompute constraint tightness based on initial positions
    initial_centers = np.column_stack([v0[0::3], v0[1::3]])
    initial_radii = v0[2::3]
    
    # Calculate constraint tightness (how close circles are to touching)
    constraint_tightness = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            dx = initial_centers[i, 0] - initial_centers[j, 0]
            dy = initial_centers[i, 1] - initial_centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            dist_to_touch = initial_radii[i] + initial_radii[j]
            constraint_tightness[i] += (dist_to_touch - dist) / dist_to_touch
            constraint_tightness[j] += (dist_to_touch - dist) / dist_to_touch
    
    # Sort indices based on constraint tightness (most constrained first)
    sorted_indices = np.argsort(constraint_tightness)[::-1]
    
    # Reorder the decision vector and constraints based on sorted indices
    reordered_v = np.zeros(3 * n)
    reordered_cons = []
    
    for i in range(n):
        idx = sorted_indices[i]
        reordered_v[3*i] = v0[3*idx]
        reordered_v[3*i+1] = v0[3*idx+1]
        reordered_v[3*i+2] = v0[3*idx+2]
        
        # Add constraints for the reordered circle
        cons_i = []
        cons_i.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_i.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_i.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_i.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        reordered_cons.extend(cons_i)
    
    # Add pairwise constraints between circles
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            reordered_cons.append({"type": "ineq", "fun": constraint_func})
    
    # Optimization with reordered constraints
    res = minimize(neg_sum_radii, reordered_v, method="SLSQP", bounds=bounds,
                   constraints=reordered_cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())