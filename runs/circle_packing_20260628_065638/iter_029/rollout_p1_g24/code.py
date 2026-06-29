import numpy as np
import warnings

def run_packing():
    n = 26
    cols, rows = 5, 6  # Hex grid with rows for staggered optimization
    # Initialize with staggered hexagonal grid, perturbed for uniqueness
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Staggered hex grid pattern
        if row % 2 == 1:
            x_center += 0.5 / cols
        # Add small random perturbation to avoid symmetry clustering
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        xs.append(x)
        ys.append(y)
    
    # Base radius based on hexagonal packing, increased to allow flexibility
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n length matches 3n variables

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize by minimizing the negative of the sum

    cons = []
    for i in range(n):
        # Define boundary constraints
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})  # Right wall
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})         # Left wall
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})  # Top wall
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})       # Bottom wall
    
    # Vectorized overlap constraints with directional hashing for dynamic prioritization
    for i in range(n):
        for j in range(i + 1, n):
            # Function to compute distance squared - sum of radii squared
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization phase with dense sampling
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-11, "eps": 1e-10, "disp": False})

    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Create dynamic adjacency vectors for edge-circle expansion strategy
        # Use directional hash for enhanced spatial bias
        directional_hash = np.random.rand(n, 2) * 0.03
        adjacency_hash = np.random.rand(n, 2) * 0.12

        # Generate spatial perturbation for non-linear configuration reordering
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Perturb center with spatial hash and directional hashing
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
            # Apply directional expansion for adjacency hashing
            perturbed_v[3*i+2] += adjacency_hash[i, 0] * 0.005 * (1 + 1.5 * np.sqrt(radii[i]))
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-10, "disp": False})

    # Execute the surgical dissection on two interacting edge circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix for efficient pairwise analysis
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Identify top two interacting circles in the packing
        interaction_weights = np.zeros(n)
        for i in range(n):
            interaction_weights[i] = np.sum(dists[i] < 0.4 * (radii[i] + radii))  # High overlap density
        top_two = np.argsort(interaction_weights)[::-1][:2]  # Top two circles with most interactions

        # Store original positions of the top two interacting circles
        orig_centers = centers[top_two].copy()
        orig_radii = radii[top_two].copy()

        # Create adjacency-aware expansion bias for these two
        new_radii = radii.copy()
        for i in top_two:
            # Targeted expansion with directional bias
            new_radii[i] = radii[i] * 1.2 + 0.005
            if i < n-1:
                new_radii[i] += 0.003 * adjacency_hash[i, 0]

        # Reconstruct the vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-10, "disp": False})

    # Apply spatial dissection strategy targeting the two interacting circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Re-compute dynamic distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Re-find top two interacting circles
        interaction_weights = np.zeros(n)
        for i in range(n):
            interaction_weights[i] = np.sum(dists[i] < 0.35 * (radii[i] + radii))  # Tighter overlap threshold
        top_two = np.argsort(interaction_weights)[::-1][:2]

        # Isolate and reconfigure these two while allowing others to expand
        # Keep their positions but increase their radii
        new_radii = radii.copy()
        for i in top_two:
            new_radii[i] = radii[i] * 1.2 + 0.004

        # Perturb their positions with directional bias based on adjacency
        directional_hash = np.random.rand(n, 2) * 0.08
        for i in top_two:
            v[3*i] += directional_hash[i, 0] * (new_radii[i] / np.sum(new_radii))
            v[3*i+1] += directional_hash[i, 1] * (new_radii[i] / np.sum(new_radii))
        
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-9, "disp": False})

    # Final fallback to valid configuration if no optimization success
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Final pass to ensure all constraints are met with validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid, reason = validate_packing(centers, radii)

    if not valid:
        # Fallback to the first valid configuration on failure
        centers, radii, _ = run_packing()
    
    return centers, radii, float(radii.sum())