import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized staggered grid and geometric perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Base positions
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Alternate row staggering
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # More refined radius initialization with geometric considerations
    r0 = 0.36 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries (with explicit index capture)
    cons = []
    for i in range(n):
        # x >= r constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # x + r <= 1 constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y >= r constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # y + r <= 1 constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing
    # Use vectorized approach to avoid O(n²) construction
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-10, "disp": False})
    
    # Asymmetric reconfiguration: random geometric hashing with constrained radius adjustment
    if res.success:
        v = res.x
        # Compute current configuration metrics
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Distance matrix for isolation detection
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find the circle with the largest minimum distance to others
        min_dists = np.min(dists, axis=1)
        isolated_idx = np.argmax(min_dists)
        
        # Compute current total sum and target expansion
        total_sum = np.sum(radii)
        expansion_factor = 0.01 / (n - 1)  # Controlled expansion to unlock new configuration
        
        # Create the new radius vector with asymmetric adjustment
        new_radii = radii.copy()
        new_radii[isolated_idx] += expansion_factor * 1.2  # Over-expansion to trigger spatial reconfiguration
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion_factor
        
        # Inject geometric randomness to avoid local minima
        random_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0] * 1.5
            perturbed_v[3*i+1] += random_hash[i, 1] * 1.5
            perturbed_v[3*i+2] = new_radii[i]  # Ensure radii are updated with expansion
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 450, "ftol": 1e-11, "disp": False})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())