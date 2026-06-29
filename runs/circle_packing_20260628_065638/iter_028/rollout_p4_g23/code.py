import numpy as np

def run_packing():
    n = 26
    cols = 6  # Optimal for 26 circles: 5x5 grid has 25, but 6 columns allows for 5 rows with 5 circles each
    
    # Staggered grid initialization with better spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / (n // cols + 1)  # Adaptive row calculation
        # Small random perturbations for breaking symmetry
        x = x_center + np.random.uniform(-0.015, 0.015)
        y = y_center + np.random.uniform(-0.015, 0.015)
        # Staggered grid offset
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initialize with slightly larger seed radii for better convergence
    r0 = 0.45 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.48)]  # Tight upper limit

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries (using index-aware lambda captures)
    cons = []
    for i in range(n):
        # Left
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints (index-aware with direct access)
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with tighter tolerances and more iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "gtol": 1e-9})

    # Apply post-optimization "shake" heuristic on least constrained circles to escape local minima
    if res.success and res.fun < 1e-9:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute minimum distance to all other circles (for constrainedness)
        min_dist_to_others = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist_to_others[i] = np.min([min_dist_to_others[i], dist])
        
        # Identify least constrained circles (those with maximum minimum distance)
        least_constrained = np.argsort(min_dist_to_others)
        
        # Apply small, targeted perturbation
        # This is a "jiggle" that helps escape shallow local minima
        perturbation = np.random.rand(3 * n)
        v_perturbed = v.copy()
        for i in range(n):
            if i in least_constrained[:4]:  # Only apply to most constrained circles
                dx = perturbation[3*i] * (radii[i] / np.mean(radii)) * 0.2
                dy = perturbation[3*i+1] * (radii[i] / np.mean(radii)) * 0.2
                dr = perturbation[3*i+2] * (radii[i] / np.mean(radii)) * 0.2
                v_perturbed[3*i] += dx
                v_perturbed[3*i+1] += dy
                v_perturbed[3*i+2] += dr

        # Re-optimization after jiggle
        res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-9})

    # Final refinement: targeted expansion with smooth gradient
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Check if expansion is possible
        min_distances = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_distances[i] = np.min([min_distances[i], dist])
        
        # Find the circle with the largest expansion potential
        expansion_idx = np.argmax(min_distances)
        max_expansion_possible = min_distances[expansion_idx] - radii[expansion_idx] - 1e-6
        if max_expansion_possible > 0:
            # Incrementally expand to avoid violating constraints
            expansion_step = max_expansion_possible * 0.8 / 10  # Conservative expansion
            for _ in range(10):
                v_new = v.copy()
                v_new[3*expansion_idx+2] += expansion_step
                centers_new = np.column_stack([v_new[0::3], v_new[1::3]])
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = centers_new[i, 0] - centers_new[j, 0]
                        dy = centers_new[i, 1] - centers_new[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < radii[i] + radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    v = v_new
                else:
                    break
        # Final refinement
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-9})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())