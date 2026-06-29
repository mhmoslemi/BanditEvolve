import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Ensure bounds list has 3*n entries for the vector of length 3n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Use functools.partial to bind i and j
            fun = lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i + 2] + v[3*j + 2])**2
            cons.append({"type": "ineq", "fun": fun})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Shake Phase: Perturb small circles only
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        sorted_indices = np.argsort(radii)
        
        # Select the smallest 5 circles to perturb
        small_circle_indices = sorted_indices[:5]
        perturbations = np.random.uniform(-0.03, 0.03, size=(2, 5))
        
        # Apply the perturbations
        for idx, (i, j) in enumerate(zip(small_circle_indices, perturbations.T)):
            v[3*i] += j[0]
            v[3*i + 1] += j[1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted expansion of the smallest circle with safe check
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Find the smallest circle
        least_constrained_idx = np.argmin(radii)
        min_radius = radii[least_constrained_idx]
        
        # Attempt to expand while keeping other circles within bounds
        # We use small expansion to avoid immediate conflict
        expansion = 0.002
        # Check if expansion is safe
        expanded_radii = radii.copy()
        expanded_radii[least_constrained_idx] += expansion
        
        # Recompute all pairwise distances (vectorized)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Check expansion feasibility
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < expanded_radii[i] + expanded_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Apply the expansion
            v[3*least_constrained_idx + 2] += expansion
            # Re-evaluate
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
        
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())