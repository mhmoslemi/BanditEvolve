import numpy as np

def run_packing():
    n = 26
    cols = 6  # increased column width for better spatial distribution
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Adjusted center calculation to account for unequal rows
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Randomized offset for fine-tuned clustering and jitter
        x_offset = np.random.uniform(-0.04, 0.04)
        y_offset = np.random.uniform(-0.04, 0.04)
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Staggered alternating rows for more even packing
        if row % 2 == 1:
            x += 0.5 / cols * (0.8 + np.random.uniform(-0.15, 0.15))
        
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation with adaptive scaling
    base_radius = 0.36 / cols - 1e-3
    # Introduce subtle radius variation to prevent cluster degeneracy
    radius_variation = np.random.uniform(0.01, 0.03, n)
    r0 = base_radius + radius_variation
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with lambda with captured i
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})

    # Vectorized overlap constraints with lambda and parameter capture
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First-pass optimization with tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-10})

    # Apply shake heuristic targeting smallest circles to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Efficient distance calculation
        dx_vecs = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy_vecs = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx_vecs**2 + dy_vecs**2)
        
        # Compute minimum distances to neighbors
        min_dists = np.min(dists, axis=1)
        # Identify the smallest circles
        least_constrained_indices = np.argsort(min_dists)[:7]  # top 7 smallest circles
        least_constrained_radii = np.sort(min_dists)[:7]  # sorted for weighted shaking

        # Shake heuristic: add jitter to circles with least freedom
        jitter_strength = 0.005  # scale per radius to ensure smaller circles shake more
        perturbation_map = np.random.rand(n, 2) * 0.05
        for idx in least_constrained_indices:
            jitter = perturbation_map[idx] * (n / least_constrained_radii + 1)
            v[3*idx] += jitter[0]
            v[3*idx+1] += jitter[1]

        # Re-optimization with shake configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})
    
    # Second optimization pass with tighter tolerances if success
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Check if at least one circle is small, if so, target expansion
        small_radius_threshold = 0.015
        small_circle_indices = np.where(radii < small_radius_threshold)[0]
        
        if len(small_circle_indices) > 0:
            # Compute potential expansion based on minimal inter-circle distances
            dist_matrix = dists.copy()
            dist_matrix = np.clip(dist_matrix, a_min=1e-10, a_max=None)
            min_dist_for_circle = np.min(dist_matrix[np.isfinite(dist_matrix)], axis=1)
            
            # Find the circle with the most potential for expansion
            potential_for_circle = min_dist_for_circle - (radii)
            best_circle_idx = np.argmax(potential_for_circle)
            best_circle_radius = radii[best_circle_idx]
            best_circle_distance = min_dist_for_circle[best_circle_idx]
            
            # Calculate expansion based on distance and current scale
            expansion_factor = (best_circle_distance - best_circle_radius) / 5
            expansion_amount = np.clip(expansion_factor * 1.2, 0, 0.005)  # safety factor
            
            # Apply expansion to this circle in a constrained way
            v[3*best_circle_idx + 2] += expansion_amount
            
            # Re-optimization after expansion attempt
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())