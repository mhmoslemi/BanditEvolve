import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Optimized base configuration: hexagonal grid with denser packing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Hexagonal grid base: even rows shift right by 0.5/cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        # Add jitter for escape from symmetry and to improve packing
        jitter = np.random.uniform(-0.08, 0.08, 2)
        x = base_x + jitter[0]
        y = base_y + jitter[1]
        # Apply row staggering for better density
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initialize radii to a higher base with careful spacing
    r_base = 0.25 / cols + 0.001  # Slightly higher than SOTA for exploration
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r_base)

    # Bounds list must match 3*n entries for the decision vector
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # (x, y, r) for each circle

    # Objective function to maximize sum of radii (negative for minimization)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraints: boundaries and overlaps
    cons = []

    # Add boundary constraints for all circles
    for i in range(n):
        # x >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Add overlap constraints: distance^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2
            })

    # First optimization phase: fine-tuning from structured initial positions
    res = minimize(neg_sum_radii, v0, method="SLSQP", 
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 750, "ftol": 1e-11, "eps": 1e-8})

    # Post-optimization refinement with geometric hashing and constraint re-evaluation
    if res.success:
        v = res.x

        # Create a geometric hash map for controlled perturbations
        # Perturb spatial parameters with radius-proportional scaling
        spatial_hash = np.random.rand(n, 2) * 0.07
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (v[3*i+2] / np.mean(v[2::3]))  # Proportional perturbation
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (v[3*i+2] / np.mean(v[2::3]))  # Proportional perturbation

        # Add radius-aware perturbations for edge-case expansion
        radii = v[2::3]
        radius_variance = radii.std()
        for i in range(n):
            if radii[i] < np.median(radii) - radius_variance * 0.5:  # Under-sized circles benefit
                perturbed_v[3*i+2] += 0.0005 * np.random.rand()  # Slight increase

        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})

    # Second phase: force reconfiguration of the two most interacted circles
    if res.success:
        v = res.x
        # Compute distances matrix for interaction analysis
        centers = np.column_stack([v[0::3], v[1::3]])
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify top two circles by total interaction strength
        interaction_strength = np.sum(dists, axis=1)
        top_idx = np.argsort(interaction_strength)[-2:]
        
        # Isolate these circles and reconfigure spatial relationship
        # Add random displacement to break previous configuration
        for i in top_idx:
            x = v[3*i] + np.random.uniform(-0.07, 0.07)
            y = v[3*i+1] + np.random.uniform(-0.07, 0.07)
            x += np.random.normal() * (v[3*i+2] / 0.5)  # Radius-based displacement
            y += np.random.normal() * (v[3*i+2] / 0.5)
            # Enforce spatial bounds
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            
            # Adjust radii to maintain spacing if needed
            r = v[3*i+2]
            new_r = r + np.random.uniform(-0.0015, 0.0015)
            new_r = np.clip(new_r, 1e-4, 0.5)
            # Create a new vector with updated position and radius
            new_v = v.copy()
            new_v[3*i] = x
            new_v[3*i+1] = y
            new_v[3*i+2] = new_r
            res = minimize(neg_sum_radii, new_v, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": 350, "ftol": 1e-11, "eps": 1e-9})
        
        v = res.x if res.success else v

    # Final step: targeted expansion of least constrained circle
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute distances and find isolating circle
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt((dx**2 + dy**2) + 1e-12)  # Prevent division by zero
        min_dists = np.min(dists, axis=1)
        isolated_idx = np.argmax(min_dists)
        
        # Calculate expansion potential based on average spacing
        avg_spacing = np.mean(min_dists)
        current_radius = radii[isolated_idx]
        max_radius = 0.5 - 0.05  # Reserves buffer
        # Compute potential expansion (based on spacing)
        expansion_potential = max((avg_spacing - current_radius) * 0.8, 0.003)
        expansion_factor = 1.0 + expansion_potential / current_radius
        
        # Create expansion vector
        new_radii = radii.copy()
        new_radii[isolated_idx] = current_radius * expansion_factor
        new_radii[isolated_idx] = np.clip(new_radii[isolated_idx], 1e-4, max_radius)
        
        # Apply expansion with gradient-based solver and constraints
        # First, perturb centers to avoid immediate overlap
        perturbed_v = v.copy()
        for i in range(n):
            if i != isolated_idx:
                perturbed_v[3*i] += np.random.uniform(-0.002, 0.002)
                perturbed_v[3*i+1] += np.random.uniform(-0.002, 0.002)
        
        # Perform optimization with expanded radii
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-9})
        
        v = res.x if res.success else v

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())