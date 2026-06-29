import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with geometric clustering, asymmetric grid, random perturbation
    xs = []
    ys = []
    rand_offset = np.random.uniform(-0.06, 0.06, size=n)
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Asymmetric staggering for better space utilization
        row_offset = 0.5 / cols * (row % 3 == 0) * np.random.choice([-1, 1]) / 3
        col_offset = 0.35 / cols * np.random.choice([-1, 1]) / 3
        
        x = x_center + col_offset + rand_offset[i]
        y = y_center + row_offset + rand_offset[i]
        
        # Prevent excessive clustering by introducing local spacing
        x = np.clip(x, 1e-5, 1 - 1e-5)
        y = np.clip(y, 1e-5, 1 - 1e-5)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radii with adaptive scaling
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with closure-based parameter binding
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with parameter closure optimization
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # First phase optimization: fine-tuning initial configuration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})

    # Forcibly isolate and reconfigure the top-most interacting circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 + 
                        (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2)
        interactivity = np.sum(dists, axis=1)
        
        # Select the top two most interconnected circles (most significant interactions)
        top_indices = np.argsort(interactivity)[-2:]
        top_centers = centers[top_indices]
        top_radii = radii[top_indices]
        
        # Create a controlled dissection zone for top two circles
        def reconfigure_dissection(v, top_indices, dists, interactivity):
            v_new = v.copy()
            # Create a controlled repulsion field for top indices
            for idx in top_indices:
                v_new[3*idx] += np.random.uniform(-0.05, 0.05)
                v_new[3*idx+1] += np.random.uniform(-0.05, 0.05)
                v_new[3*idx+2] += np.random.uniform(-0.004, 0.004)
            
            # Enforce new non-overlap
            for i in top_indices:
                for j in top_indices:
                    if i != j:
                        dx = v_new[3*i] - v_new[3*j]
                        dy = v_new[3*i+1] - v_new[3*j+1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < top_radii[i] + top_radii[j] - 1e-10:
                            # Adjust radius for non-overlap
                            v_new[3*i+2] = max(v_new[3*i+2], dist - top_radii[j] + 1e-10)
                            v_new[3*j+2] = max(v_new[3*j+2], dist - top_radii[i] + 1e-10)
            return v_new
        
        v = reconfigure_dissection(v, top_indices, dists, interactivity)
        # Re-verify after dissection
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Introduce controlled forced adjacency constraint
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Find the next best circle to introduce a forced adjacency to the top circle
        dists = np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 + 
                        (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2)
        
        # Identify the two closest centers (excluding the top one)
        top_index = top_indices[0]
        dists_top = dists[top_index]
        min_dists = np.argsort(dists_top)
        min_index = min_dists[1] if top_index != min_dists[1] else min_dists[2]
        
        # Force adjacency between the two circles (with minimal radius)
        forced_distance = radii[top_index] + radii[min_index] + 0.001
        dx = centers[top_index, 0] - centers[min_index, 0]
        dy = centers[top_index, 1] - centers[min_index, 1]
        norm = np.sqrt(dx**2 + dy**2)
        
        if norm == 0:
            # Prevent zero distance case
            dx, dy = np.random.uniform(-0.01, 0.01, size=2)
            norm = np.sqrt(dx**2 + dy**2)
        
        # Create forced movement vector with controlled radius expansion
        movement = (forced_distance / norm - 1.0) * np.array([dx, dy])
        v_new = v.copy()
        v_new[3*top_index] += movement[0]
        v_new[3*min_index] += movement[1]        
        # Enforce bounds
        v_new[3*top_index] = np.clip(v_new[3*top_index], 1e-4, 1 - 1e-4)
        v_new[3*min_index] = np.clip(v_new[3*min_index], 1e-4, 1 - 1e-4)        
        # Adjust radii to maintain non-overlap
        v_new[3*top_index+2] = max(v_new[3*top_index+2], forced_distance - v_new[3*min_index+2] + 1e-12)
        v_new[3*min_index+2] = max(v_new[3*min_index+2], forced_distance - v_new[3*top_index+2] + 1e-12)
        
        # Re-evaluate with new constraint
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    # Targeted spatial dissection to optimize under-utilized area
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 + 
                        (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2)
        
        # Calculate utilization metric: distance to nearest circle / diameter
        utilization_metric = np.zeros(n)
        for i in range(n):
            for j in range(i+1, n):
                utilization_metric[i] += (dists[i, j] / (radii[i] + radii[j])) ** 2
                utilization_metric[j] += (dists[j, i] / (radii[j] + radii[i])) ** 2
        utilization_metric /= n
        
        # Select the circle with the lowest utilization to expand
        least_utilized_idx = np.argmin(utilization_metric)
        # Expand this circle with careful spatial adjustment
        
        # Apply adaptive expansion while enforcing non-overlap
        growth_factor = 0.005  # Maximum radius growth
        new_radius = radii[least_utilized_idx] + growth_factor * np.random.uniform(0.9, 1.1)
        
        # Ensure expansion does not violate constraints
        while True:
            v_new = v.copy()
            v_new[3*least_utilized_idx + 2] = new_radius
            centers_new = np.column_stack([v_new[0::3], v_new[1::3]])
            radii_new = v_new[2::3]
            
            # Check for all overlapping
            overlap = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = centers_new[i, 0] - centers_new[j, 0]
                    dy = centers_new[i, 1] - centers_new[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < radii_new[i] + radii_new[j] - 1e-10:
                        overlap = True
                        break
                if overlap:
                    break
            
            if not overlap:
                # Accept expansion
                break
            else:
                # Reduce expansion slightly
                new_radius = radii[least_utilized_idx] + (growth_factor * 0.9) * np.random.uniform(0.9, 1.1)
        
        v = v_new
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())