import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols

    # Initialize positions with adaptive spatial hashing for dynamic density control
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add structured perturbation based on spatial position and row parity
        # This creates dynamic clustering while preserving global reachability
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        if row % 2 == 1:
            x += 0.5 / cols * (0.5 + np.random.uniform(-0.5, 0.5))
        else:
            x += 0.5 / cols * (0.25 + np.random.uniform(-0.25, 0.25))
        if col % 2 == 1:
            y += 0.5 / rows * (0.25 + np.random.uniform(-0.25, 0.25))
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Create bounds with strict spatial and radius constraints
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n for 3*26 variables

    def neg_sum_radii(v):
        # Ensure we don't prematurely clip to avoid gradient issues during intermediate optimization
        return -np.sum(v[2::3])

    # Optimized constraint construction with explicit capture and vectorization
    cons = []

    # Add boundary constraints (left, right, bottom, top)
    for i in range(n):
        # Left - radius >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right - radius <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom - radius >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top - radius <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Add pairwise distance constraints with adaptive scaling for efficiency
    # Use vectorized computation in constraint functions
    for i in range(n):
        for j in range(i+1, n):
            # Use a more efficient constraint function with direct access to indices
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j:
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                    - (v[3*i+2] + v[3*j+2])**2
            })

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, 
                   options={"maxiter": 800, "ftol": 1e-11, "gtol": 1e-10})
    
    # Adaptive post-optimization with targeted spatial and radius exploration
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # First: Perturb the spatial configuration of most constrained circles
        # Identify circles with minimal min distance (most constrained)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                distances[i, j] = np.sqrt(dx*dx + dy*dy)
                distances[j, i] = distances[i, j]
        
        # Compute minimum distances and find the two most constrained circles
        min_dists = np.min(distances, axis=1)
        idxs = np.argsort(min_dists)[:2]
        i, j = idxs
        
        # Apply asymmetric spatial perturbation
        # Shift i-circle away to reduce overlap
        perturbation = np.random.rand(2) * 0.04 * (radii[i] + radii[j])
        v[3*i] += perturbation[0]
        v[3*j] += perturbation[1]
        
        # Second: Apply radius expansion heuristic with global constraint checking
        # This is an explicit expansion pass, not just a perturbation
        new_radii = radii.copy()
        # Estimate target radius growth based on current sum and spatial capacity
        # We'll expand by 0.0075 while maintaining non-overlap
        expansion_budget = 0.0075
        
        # Apply expansion to all circles, but prioritize those with the most space
        # Recompute minimal distances to avoid overlaps
        for _ in range(2):  # Two attempts to converge
            new_centers = np.column_stack([v[0::3], v[1::3]])
            new_radii = new_radii.copy()
            
            # Attempt to expand all radii proportionally to their spatial freedom
            # Compute for each circle: max expansion possible based on closest neighbor
            safe_radius_expansions = []
            for ci in range(n):
                max_expansion = 0.0
                for cj in range(n):
                    if cj == ci:
                        continue
                    d = np.sqrt((new_centers[ci, 0] - new_centers[cj, 0])**2 + 
                               (new_centers[ci, 1] - new_centers[cj, 1])**2)
                    if d < 1e-8:
                        max_expansion = 0.0
                        break
                    max_expansion_val = (d - new_radii[ci] - new_radii[cj])/2
                    if max_expansion_val < max_expansion:
                        max_expansion = max_expansion_val
                safe_radius_expansions.append(max_expansion)
            
            # Normalize expansions to be within budget
            expansion_factors = [safe_radius_expansions[i] / np.max(safe_radius_expansions) for i in range(n)]
            expansion_factors = [min(ef, 1.0) for ef in expansion_factors]
            
            # Apply expansion
            expansion_vector = np.array([expansion_factors[i] * expansion_budget for i in range(n)])
            new_radii += expansion_vector
            
            # Now, validate these new radii
            valid = True
            for ci in range(n):
                for cj in range(n):
                    if ci >= cj:
                        continue
                    d = np.sqrt((new_centers[ci, 0] - new_centers[cj, 0])**2 + 
                               (new_centers[ci, 1] - new_centers[cj, 1])**2)
                    if d < new_radii[ci] + new_radii[cj] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # If invalid, halve the expansion budget to preserve feasibility
                expansion_budget *= 0.5

        # Apply valid expansion to the decision vector
        v[2::3] = np.clip(new_radii, 1e-4, 0.5)
        
        # Update the decision vector and restart optimizer with this expanded configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, 
                       options={"maxiter": 800, "ftol": 1e-11, "gtol": 1e-10})
    
    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    return centers, radii, float(radii.sum())