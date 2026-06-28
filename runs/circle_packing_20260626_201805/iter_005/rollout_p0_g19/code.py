import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorize the overlap constraints for better performance
    def vectorized_overlap_constraint(v):
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        dx = x_centers[:, np.newaxis] - x_centers[np.newaxis, :]
        dy = y_centers[:, np.newaxis] - y_centers[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r_radii[:, np.newaxis] + r_radii[np.newaxis, :])**2
        return dist_sq - min_dist_sq

    # Convert to list of functions for each pair
    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            overlap_cons.append({"type": "ineq", "fun": constraint_func})

    cons.extend(overlap_cons)

    # Apply a dual-phase mutation strategy with topological reconfiguration and forced subcomponent optimization
    def geometric_distortion(v):
        # Apply a random rotation and scaling to the initial guess
        theta = np.random.uniform(-np.pi/4, np.pi/4)
        scale = np.random.uniform(0.8, 1.2)
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        
        # Rotate and scale the positions
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotated_x = x_centers * cos_theta - y_centers * sin_theta
        rotated_y = x_centers * sin_theta + y_centers * cos_theta
        distorted_x = rotated_x * scale
        distorted_y = rotated_y * scale
        
        # Apply clipping to ensure bounds are respected
        distorted_v = np.zeros_like(v)
        distorted_v[0::3] = np.clip(distorted_x, 0.0, 1.0)
        distorted_v[1::3] = np.clip(distorted_y, 0.0, 1.0)
        distorted_v[2::3] = r_radii
        return distorted_v

    # Add a localized perturbation of the smallest radius circles
    def localized_perturbation(v):
        # Identify small circles (radii < 0.1)
        small_circle_indices = np.where(v[2::3] < 0.1)[0]
        if len(small_circle_indices) > 0:
            # Perturb their positions slightly
            perturbation = 0.05 * np.random.rand(3 * n)
            v_perturbed = v + perturbation
            # Apply clipping to ensure bounds are respected
            v_perturbed[0::3] = np.clip(v_perturbed[0::3], 0.0, 1.0)
            v_perturbed[1::3] = np.clip(v_perturbed[1::3], 0.0, 1.0)
            v_perturbed[2::3] = np.clip(v_perturbed[2::3], 1e-4, 0.5)
            return v_perturbed
        return v

    def topological_reconfiguration(v):
        # Split into clusters and permute them
        cluster_size = 5
        clusters = []
        for i in range(0, n, cluster_size):
            cluster = v[3*i:3*(i+cluster_size)]
            clusters.append(cluster)
        
        # Shuffle clusters
        np.random.shuffle(clusters)
        reconfigured_v = np.concatenate(clusters)
        
        # Ensure bounds are respected after reconfiguration
        reconfigured_v[0::3] = np.clip(reconfigured_v[0::3], 0.0, 1.0)
        reconfigured_v[1::3] = np.clip(reconfigured_v[1::3], 0.0, 1.0)
        reconfigured_v[2::3] = np.clip(reconfigured_v[2::3], 1e-4, 0.5)
        return reconfigured_v

    # Apply the topological reconfiguration to the initial guess
    v_distorted = geometric_distortion(v0)
    v_perturbed = localized_perturbation(v_distorted)
    v_reconfigured = topological_reconfiguration(v_perturbed)
    
    # Apply forced subcomponent optimization
    def forced_subcomponent_optimization(v):
        # Partition the layout into subcomponents
        subcomponent_size = 5
        subcomponents = []
        for i in range(0, n, subcomponent_size):
            subcomponent = v[3*i:3*(i+subcomponent_size)]
            subcomponents.append(subcomponent)
        
        # Ensure at least one subcomponent has increased average radius by 5%
        avg_radii = [np.mean(subcomponent[2::3]) for subcomponent in subcomponents]
        max_avg_radius = np.max(avg_radii)
        target_avg_radius = max_avg_radius * 1.05
        for i in range(len(subcomponents)):
            subcomponent = subcomponents[i]
            if np.mean(subcomponent[2::3]) < target_avg_radius:
                # Perturb this subcomponent slightly
                perturbation = 0.05 * np.random.rand(3 * subcomponent_size)
                modified_subcomponent = subcomponent + perturbation
                # Clip to ensure bounds
                modified_subcomponent[0::3] = np.clip(modified_subcomponent[0::3], 0.0, 1.0)
                modified_subcomponent[1::3] = np.clip(modified_subcomponent[1::3], 0.0, 1.0)
                modified_subcomponent[2::3] = np.clip(modified_subcomponent[2::3], 1e-4, 0.5)
                subcomponents[i] = modified_subcomponent
        
        # Reassemble the layout
        optimized_v = np.concatenate(subcomponents)
        return optimized_v

    v_forced = forced_subcomponent_optimization(v_reconfigured)
    
    # Run the optimization with reconfigured initial guess
    res = minimize(neg_sum_radii, v_forced, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())