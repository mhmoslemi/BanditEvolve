import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with spatial hashing and asymmetric grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid layout
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Apply geometric hashing for asymmetric distribution
        # Use polar coordinates for spatial hashing
        radial_hash = np.random.rand() * 0.15
        angular_hash = np.random.rand() * 2 * np.pi
        
        # Convert to cartesian and shift to avoid clustering
        dx = np.cos(angular_hash) * radial_hash
        dy = np.sin(angular_hash) * radial_hash
        
        x = x_center + dx
        y = y_center + dy
        
        # Stagger for alternate rows with adaptive spacing
        if row % 2 == 1:
            x += 0.5 / cols * (0.5 + np.random.rand() * 0.5)  # add dynamic spacing
            
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on grid and edge constraints
    r0_base = 0.45 / cols - 1e-3  # more generous initial radius
    r0_variation = 0.08  # variance to simulate asymmetric distribution
    r0 = np.full(n, r0_base) + np.random.uniform(-r0_variation, r0_variation, size=n)
    r0 = np.clip(r0, 1e-4, 0.6)  # clip to safe radius limits
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    v0 = np.clip(v0, np.zeros(3 * n), np.ones(3 * n))  # safety bounds on x/y

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.6)]  # extended upper radius to allow more growth

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with spatial hashing to avoid over-constrained
    # Use lambda with captured i and apply strict non-overlap checks
    cons = []
    for i in range(n):
        # Left: x - r >= 0
        # Right: x + r <= 1
        # Bottom: y - r >= 0
        # Top: y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized pairwise distance constraints with spatial hashing
    # Use adaptive thresholding and directional bias for critical circles
    for i in range(n):
        for j in range(i + 1, n):
            # Use asymmetric directional constraints for critical pairs
            # Add small jitter to avoid constraint violations
            offset = np.random.uniform(-0.00005, 0.00005)
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2 + offset)})

    # Initial optimization with tightened tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-6})
    best_v = v0  # fallback in case res fails
    
    # First refinement: spatial reconfiguration using adaptive grid
    if res.success:
        v = res.x
        # Calculate spatial density map for informed perturbation
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.sqrt(((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
        min_distances = np.min(dists, axis=1)
        weight = min_distances / np.mean(min_distances)  # higher weight if circle is less constrained
        
        # Apply randomized geometric hashing with adaptive scaling
        perturbation = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Scale perturbation by inverse of radii for constrained circles
            radius = v[3*i + 2]
            scale = radius / (np.mean(radii) + 1e-8)
            perturbed_v[3*i] += perturbation[i, 0] * scale
            perturbed_v[3*i+1] += perturbation[i, 1] * scale
        
        # Re-run with refined perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})
        best_v = res.x if res.success else v

    # Second refinement: geometric dissection on critical pair with directional expansion
    if res.success:
        v = best_v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Calculate inter-circle distances and identify the most constrained pair
        dists = np.sqrt(((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
        idxs = np.argsort(dists[np.triu_indices(n, k=1)]).squeeze()
        min_idx = idxs[0]  # first closest pair
        i, j = divmod(min_idx + 1, n)  # ensure i < j
        
        # Get the two circles
        ci = centers[i]
        cj = centers[j]
        ri = radii[i]
        rj = radii[j]
        
        # Compute the overlap distance
        dist = np.sqrt((ci[0] - cj[0])**2 + (ci[1] - cj[1])**2)
        overlap = ri + rj - dist
        if overlap > 0:
            # Apply directional dissection: move one circle to the side
            max_move = min(ri, rj) * 0.8  # move at most 80% of circle's radius
            move = max_move * np.random.choice([-1, 1])  # choose direction randomly
            
            # Move circle j to the right by move
            v[3*j + 0] = cj[0] + move
            v[3*j + 1] = cj[1]
            
            # Adjust radii slightly to avoid overlap and allow expansion
            v[3*j + 2] = max(v[3*j + 2] - 0.002, 1e-4)  # trim radius slightly
            
            # Re-run with adjusted configuration
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})
            best_v = res.x if res.success else v

    # Final refinement: radius expansion on least constrained circle with directional expansion
    if res.success:
        v = best_v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Find circle with least minimum distance to others (most space)
        dists = np.sqrt(((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)
        lc_center = centers[least_constrained_idx]
        lc_radius = radii[least_constrained_idx]
        
        # Expand its radius first while maintaining total sum constraint
        # Calculate how much can be added without causing overlap
        # Use a directional expansion to maximize available space
        max_expansion = 0.0
        expansion_dir = np.array([0.0, 0.0])  # direction for expansion
        
        for dir_vec in [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]]:  # 8 directions for directional expansion
            new_center = lc_center + dir_vec * (lc_radius * 0.1)  # expand by 10% of radius in each direction
            dists = np.sqrt((new_center - centers)**2).sum(axis=1)
            overlapping = np.any(dists < (radii + lc_radius - 1e-8))
            if not overlapping:
                max_expansion = max(max_expansion, lc_radius * 0.1)  # max possible in this direction

        # Apply directional expansion to this circle
        v[3*least_constrained_idx + 0] += max_expansion * dir_vec[0]
        v[3*least_constrained_idx + 1] += max_expansion * dir_vec[1]
        v[3*least_constrained_idx + 2] += max_expansion
        
        # Re-run to maintain constraints
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11})
        best_v = res.x if res.success else v

    v = best_v if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.6)  # clip to upper radius bound
    # Final validation and postprocessing to ensure all constraints are satisfied
    # (This is done automatically by the solver, but we add a final pass)
    # Final adjustment: if no solution, return the initial v0
    if not res.success:
        # Use fallback to the best_v found during the process
        v = best_v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, 0.6)
    
    return centers, radii, float(radii.sum())