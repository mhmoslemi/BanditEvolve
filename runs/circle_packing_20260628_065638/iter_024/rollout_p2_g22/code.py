import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Generate asymmetric spatial hash with random perturbations
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Asymmetric perturbation: use row and col values to seed randomization
        seed = (row * 100 + col) % 1000
        x_offset = np.random.RandomState(seed).uniform(-0.1, 0.1)
        y_offset = np.random.RandomState(seed).uniform(-0.1, 0.1)
        
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Stagger alternate rows for a more irregular layout
        if row % 2 == 1:
            x += 0.4 / cols
        
        xs.append(x)
        ys.append(y)
    
    # Initial radii: use a smaller radius to allow expansion
    r0 = 0.3 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraint functions
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized pairwise distance constraint (non-overlap)
    for i in range(n):
        for j in range(i + 1, n):
            # We use vectorized calculation with broadcasting
            def distance_constraint(v, i=i, j=j):
                # Vectorized approach with broadcasting
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": distance_constraint})
    
    # First optimization with initial configuration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
    
    # Asymmetric spatial disruption: apply large-scale geometric hashing
    if res.success:
        v = res.x
        # Add asymmetric perturbations using row and col information
        perturb = np.zeros_like(v)
        for i in range(n):
            row = i // cols
            col = i % cols
            seed = (row * 1000 + col) % 1000
            r = np.random.RandomState(seed).uniform(-0.08, 0.08)
            perturb[3*i] += r
            perturb[3*i+1] += r
            perturb[3*i+2] += np.random.RandomState(seed).uniform(-0.008, 0.008)
        perturbed_v = v + perturb
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Vectorized pairwise distance matrix
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Identify most under-constrained circle (least interaction with neighbors)
        interaction = np.sum(1 / (dists + 1e-6), axis=1)
        most_unguided_idx = np.argmin(interaction)
        
        # Calculate expansion factor based on current radius and potential free space
        expansion_factor = 0.008 / (n - 1)  # Controlled expansion to unlock new configurations
        
        # Apply expansion while maintaining non-overlap constraints
        new_radii = radii.copy()
        new_radii[most_unguided_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != most_unguided_idx:
                new_radii[i] += expansion_factor
        
        # Update and optimize
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())