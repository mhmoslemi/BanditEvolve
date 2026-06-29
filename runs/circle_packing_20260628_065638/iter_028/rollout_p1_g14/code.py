import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Seed for deterministic behavior during iteration
    np.random.seed(42)

    # Step 1: Generate initial grid with adaptive clustering and spatial hashing
    xs = np.zeros(n)
    ys = np.zeros(n)
    radius_guesses = np.zeros(n)

    # Step 1a: Generate base grid in staggered lattice
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows

        # Step 1b: Add random perturbation with decaying intensity
        x_offset = np.random.uniform(-0.05, 0.05) / (1 + row)  # decaying perturbation
        y_offset = np.random.uniform(-0.05, 0.05) / (1 + row + 0.5) * (1.0 if row % 2 == 0 else 1.2)
        x = x_center + x_offset
        y = y_center + y_offset

        # Step 1c: Apply staggered offset for alternate rows
        if row % 2 == 1:
            x += 0.5 / cols * (1.1 if row % 3 == 1 else 1.0)

        xs[i] = x
        ys[i] = y

    # Step 1d: Initial radius guess based on grid spacing and spacing coefficient
    min_grid_dist = 0.8 + (0.9 - 0.8) * (0.01)  # 80% to 90% of max unit distance
    min_dist = 0.5 * (np.min(0.5 / cols + 0.1) + 0.05)  # Adjust for clustering
    r0 = 0.25 * (min_grid_dist - 0.005) * (1.0 + (np.random.rand() * 0.1 - 0.05))  # Small random variability
    radius_guesses[:] = r0

    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = radius_guesses

    # Step 2: Define bounds with precise alignment
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Step 3: Define objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Step 4: Define boundary constraints
    cons = []
    for i in range(n):
        # Ensure x - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Ensure x + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Ensure y - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Ensure y + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})

    # Step 5: Define pairwise overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Lambda capture fixed with lambda i,j: (lambda v: ...)
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j: 
                        (v[3*i] - v[3*j])**2 + 
                        (v[3*i+1] - v[3*j+1])**2 -
                        (v[3*i+2] + v[3*j+2])**2)
            })

    # Step 6: First optimization pass with fine-tuned solver options
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "eps": 1e-8})

    # Step 7: Apply radical reconfiguration via geometric tiling and spatial hashing
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Create spatial and adjacency hashings with adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.04
        adjacency_hash = np.random.rand(n, 2) * 0.06
        
        # Step 7a: Spatial reconfiguration with directional hashing
        v_perturbed = v.copy()
        for i in range(n):
            v_perturbed[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) * 1.2
            v_perturbed[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii)) * 1.2
            
            if i < n - 2:
                v_perturbed[3*i+2] += adjacency_hash[i, 0] * 0.003
                v_perturbed[3*i+1] += adjacency_hash[i, 1] * 0.002

        res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9})
    
    # Step 8: Apply targeted radius expansion on least constrained circle with soft constraints
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Vectorized distance computation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(np.sort(min_dists))  # Actually find the one with most slack
        
        # Step 8a: Compute growth potential based on total sum
        current_total = np.sum(radii)
        target_growth = 0.006  # 0.6% increase goal
        expansion_factor = target_growth / (n - 0.999) * (current_total / np.sum(radii))
        
        # Step 8b: Directional expansion with stochastic and adjacency-aware enhancements
        directional_hash = np.random.rand(n, 2) * 0.05
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.15  # Over-expansion factor
        for i in range(n):
            if i != least_constrained_idx:
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                if adj_weight < 0.1:
                    expansion = expansion_factor * 1.25
                else:
                    expansion = expansion_factor * (1.0 + 0.5 * np.random.rand())
                new_radii[i] += expansion
        
        # Step 8c: Apply expansion with constraint validation and smart scaling
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion with decay
                new_radii = radii + (new_radii - radii) * 0.98
        
        # Final optimization with reconfigured radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-9})
    
    # Final fallback to initial solution if optimization fails
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final validation of positions and radii
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12 or
            y - r < -1e-12 or y + r > 1 + 1e-12):
            radii[i] = min(r, max(0.0, 1.0 - np.max([x, y])) * 0.5)
    
    return centers, radii, float(radii.sum())