import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with staggered grid and randomized offsets
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.03, 0.03)
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.37 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for the decision vector (length 3n)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective: maximize sum of radii by minimizing negative sum
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Add boundary constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Add pairwise circle overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Initial global optimization with tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "eps": 1e-10})
    
    if res.success:
        # Generate geometric hash to disrupt local minima
        geometric_hash = np.random.rand(n, 2) * 0.04 * (np.sqrt(np.sum(res.x[2::3]**2)))
        perturbed_v = res.x.copy()
        for i in range(n):
            perturbed_v[3*i] += geometric_hash[i, 0]
            perturbed_v[3*i+1] += geometric_hash[i, 1]
        
        # Reoptimize with new perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-10})
    
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Find the circle with smallest radius (least constrained)
        smallest_radius_idx = np.argmin(radii)
        smallest_radius = radii[smallest_radius_idx]
        
        # Calculate spatial expansion vector with adaptive scaling
        expansion_vec = np.zeros(n)
        expansion_vec[smallest_radius_idx] = np.random.uniform(0.0008, 0.0014)
        expansion_vec += 0.0002 * np.random.rand(n)
        
        # Targeted refinement with soft constraint relaxation
        for _ in range(2):
            # Create expansion vector with gradual increase
            expanded_v = v.copy()
            expanded_v[2::3] += expansion_vec
            
            # Evaluate validity
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx = expanded_v[3*i] - expanded_v[3*j]
                    dy = expanded_v[3*i+1] - expanded_v[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < (expanded_v[3*i+2] + expanded_v[3*j+2]) - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Accept expansion if valid
                v = expanded_v
                break
            else:
                # Reduce expansion vector by 25% if invalid
                expansion_vec *= 0.75
        
        # Final optimization after targeted expansion
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-11})
    
    # Post-processing
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())