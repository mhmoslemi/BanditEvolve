import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with staggered grid and fine-tuned randomness
    def generate_grid():
        xs = []
        ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            x_center = (col + 0.5) / cols
            y_center = (row + 0.5) / rows
            # Create fine-tuned random jitter to avoid collisions while preserving structure
            x = x_center + np.random.uniform(-0.05, 0.05)
            y = y_center + np.random.uniform(-0.05, 0.05)
            # Stagger alternate rows for density control
            if row % 2 == 1:
                x += 0.5 / cols * (0.8 + np.random.uniform(-0.1, 0.1))
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    xs, ys = generate_grid()
    # Initial radii based on space allocation and safety margin
    r0 = 0.35 / cols * (1 + 0.05 * np.random.rand(n)) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with lambda captures
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with dynamic lambda capture
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                         - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with adaptive max iterations and enhanced tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})

    # Apply "shake" heuristic - perturb smallest circles to escape local optima
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find smallest radius with safety margin
        smallest_radius_idx = np.argmin(radii)
        smallest_radius = radii[smallest_radius_idx]
        # If smallest is not below 0.01, we risk being too tight
        if smallest_radius < 0.01:
            # Perturb the coordinate by 2% of the square edge to increase space
            perturb = np.random.uniform(-0.02, 0.02, size=2)
            perturbed_v = v.copy()
            perturbed_v[3*smallest_radius_idx] += perturb[0]
            perturbed_v[3*smallest_radius_idx + 1] += perturb[1]
            # Add 2% radius growth to small circles
            perturbed_v[3*smallest_radius_idx + 2] = min(radii[smallest_radius_idx] * 1.02, 0.5)
            # Re-run optimization after shake
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Final optimization with controlled expansion and reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        # Vectorized distance calculation with optimized broadcasting
        dx = v[0::3, np.newaxis] - v[0::3]
        dy = v[1::3, np.newaxis] - v[1::3]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle (max minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Expand radii in a controlled manner using geometric hashing
        max_radius = 0.5
        expansion_factor = 0.006 / np.sum(radii) * (max_radius / np.mean(radii))
        expansion_vector = np.zeros(n)
        expansion_vector[least_constrained_idx] = 1.2 * expansion_factor
        for i in range(n):
            if i != least_constrained_idx:
                expansion_vector[i] = expansion_factor * (1.0 + 0.2 * np.random.rand())
        
        # Apply expansion with feasibility checks
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] += expansion_vector
            expanded_v[2::3] = np.clip(expanded_v[2::3], 1e-4, 0.5)
            
            # Validate the expanded configuration for non-overlap
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_v[3*i] - expanded_v[3*j]
                    dy = expanded_v[3*i+1] - expanded_v[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < (expanded_v[3*i+2] + expanded_v[3*j+2]) - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion proportionally if invalid
                expansion_vector /= 2
        
        # Re-run with fine-tuned parameters
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())