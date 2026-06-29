import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with geometric hashing and adaptive placement
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add geometric hashing to spread out the initialization
        hash_seed = np.random.rand(2)
        x = x_center + 0.02 * hash_seed[0] * np.cos(2.0 * np.pi * row / cols)
        y = y_center + 0.02 * hash_seed[1] * np.sin(2.0 * np.pi * col / rows)
        # Apply conditional stagger
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with adaptive scaling based on cols and rows
    avg_radius = 0.35 / cols
    r0 = avg_radius - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Ensure bounds match decision vector length (3 * n)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        # x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # 1 - (x + r) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # 1 - (y + r) >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Precompute the constraint in lambda with captured i,j
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # First optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "eps": 1e-8})
    
    # Apply shake heuristic if optimization was successful
    if res.success:
        v = res.x
        # Perform multiple shake iterations
        for shake_iter in range(3):
            # Compute radii vector
            radii = v[2::3]
            # Compute min radius and its index
            min_radius = np.min(radii)
            least_constrained_idx = np.argmin(radii)
            
            # Jiggle the least constrained circle
            perturb_scale = 1e-4 * np.sqrt(min_radius)  # Scale perturbation with radius size
            for i in range(n):
                if i == least_constrained_idx:
                    # Add small random perturbation to position and radius
                    v[3*i] += np.random.uniform(-perturb_scale, perturb_scale)
                    v[3*i+1] += np.random.uniform(-perturb_scale, perturb_scale)
                    v[3*i+2] += np.random.uniform(-0.0005, 0.0005)  # Adjust radius
                else:
                    # Small perturbation to other circles for diversity
                    v[3*i] += np.random.uniform(-1e-6, 1e-6)
                    v[3*i+1] += np.random.uniform(-1e-6, 1e-6)
            
            # Re-optimize with small steps
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 200, "ftol": 1e-10, "eps": 1e-8})
            v = res.x

    # Final optimization pass to refine
    if res.success:
        v = res.x
        # Final targeted expansion - distributed expansion with feedback
        radii = v[2::3]
        total_sum = np.sum(radii)
        # Compute the average circle spacing to determine growth potential
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        growth_ratio = np.max(min_dists) / (np.max(radii) + np.mean(radii)) * 1.2
        
        # Targeted expansion with dynamic scaling
        radii_new = radii * (1.0 + growth_ratio * (1.0 - 0.9))
        v_new = v.copy()
        v_new[2::3] = np.clip(radii_new, 1e-6, 0.5)
        
        # Final re-evaluation with expanded radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-8})
        v = res.x

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())