import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    def initialize_positions(n, cols, rows):
        # Create a grid of cluster centers
        cluster_centers = []
        for i in range(rows):
            for j in range(cols):
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                cluster_centers.append((x, y))
        # Randomly select a subset of cluster centers to form the initial positions
        initial_positions = np.array(cluster_centers)[np.random.choice(len(cluster_centers), size=n, replace=False)]
        # Add small random perturbation to break symmetry
        initial_positions += np.random.uniform(-0.05, 0.05, size=initial_positions.shape)
        return initial_positions
    
    # Generate initial positions using the clustering approach
    initial_positions = initialize_positions(n, cols, rows)
    xs = initial_positions[:, 0]
    ys = initial_positions[:, 1]
    
    r0 = 0.3 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

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
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply a controlled radius expansion to the most isolated cluster
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Compute pairwise distances between centers
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.sqrt(np.sum((centers[i] - centers[j])**2))
        # Find the most isolated cluster (largest minimum distance to others)
        min_distances = np.min(dist_matrix, axis=1)
        isolated_index = np.argmax(min_distances)
        # Increase the radius of the isolated circle by a small amount
        perturbed_v = v.copy()
        perturbed_v[3*isolated_index+2] += 0.01
        # Clip radii to ensure they stay within bounds
        perturbed_v[2::3] = np.clip(perturbed_v[2::3], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())