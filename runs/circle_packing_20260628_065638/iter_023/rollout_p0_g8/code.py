import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = n // cols + (1 if n % cols else 0)
    
    # Randomized geometric hashing initialization
    hash_seed = np.random.rand(n, 2)
    xs = (hash_seed[:, 0] * cols) % 1.0
    ys = (hash_seed[:, 1] * rows) % 1.0
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints using lambda with i as closure
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
                return dx**2 + dy**2 - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12})

    # Apply geometric hashing hybrid reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create spatial hash for reconfiguration
        hash_seed = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_seed[i, 0]
            perturbed_v[3*i+1] += hash_seed[i, 1]
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Controlled radius expansion on smallest circle with total sum constraint
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate current radius sum and determine expansion amount
        current_sum = np.sum(radii)
        target_sum = current_sum + 0.008
        
        # Find the smallest circle with non-zero radius
        non_zero_mask = radii > 1e-6
        if np.any(non_zero_mask):
            smallest_idx = np.argmin(radii[non_zero_mask])
            expansion_per_circle = (target_sum - current_sum) / n
            new_radii = radii.copy()
            new_radii[np.where(non_zero_mask)] += expansion_per_circle * 1.1  # Small over-expectation
            
            # Update the decision vector
            v_new = v.copy()
            v_new[2::3] = new_radii
            
            # Re-run optimization with new radii
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())