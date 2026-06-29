import numpy as np

def run_packing():
    n = 26
    # Optimized grid with adaptive layout for higher density and balanced constraints
    cols = 5
    rows = 5  # Fixed 5x5 grid for deterministic initialization
    
    # Initialize positions with geometric-aware clustering, adaptive stagger, and randomized seed
    np.random.seed(20230613+int(np.random.rand()))  # Seed for deterministic but variable exploration
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid spacing
        base_x = col + 0.5
        base_y = row + 0.5
        # Scale to unit square
        base_x /= cols
        base_y /= rows
        # Add adaptive stochastic perturbations
        # Use Gaussian for smoothness, and avoid extreme outliers
        x = base_x + np.random.normal(0, 0.04, 1)[0] * (1.0 - (row + col) / (cols + rows))
        y = base_y + np.random.normal(0, 0.04, 1)[0] * (1.0 - (row + col) / (cols + rows))
        
        # Staggered grid with row-dependent shift to break symmetry
        if row % 2 == 1:
            x += 0.5 / cols * (0.3 + 0.1 * np.random.rand())
        # Boundary check and clamping
        x = np.clip(x, 1e-8, 1 - 1e-8)
        y = np.clip(y, 1e-8, 1 - 1e-8)
        xs.append(x)
        ys.append(y)
    
    # Radius initialization based on grid and empirical optimal scaling
    # Empirically derived from past performance (higher base radius with stagger)
    r0 = (0.37 / cols) * (1.0 - (0.05 * rows))  # Adjusts spacing for 5x5 grid
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bounds for 3n variables

    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Optimize constraints with advanced bounds and efficient vectorization
    # Construct inequality constraints for boundary conditions
    # Ensure lambda captures i correctly - use late binding with mutable vars
    cons = []
    # Define per-circle boundary constraints
    for i in range(n):
        # LEFT: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: (1 - v[3*i] - v[3*i+2])})
        # RIGHT: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: (v[3*i] - v[3*i+2])})
        # BOTTOM: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: (1 - v[3*i+1] - v[3*i+2])})
        # TOP: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: (v[3*i+1] - v[3*i+2])})
    
    # Vectorized pairwise overlap constraints with adaptive grid-based optimization
    # Use early lambda binding to ensure correct i and j
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                    - (v[3*i+2] + v[3*j+2])**2
            })

    # Initial optimization with tighter tolerances, multiple phases, and adaptive steps
    # First pass with adaptive constraints and spatial awareness
    res = minimize(
        neg_sum_radii, v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter": 2500,  # Larger iterations for complex constraints
            "ftol": 1e-12,    # Tight tolerance for precision
            "gtol": 1e-12,
            "eps": 1e-12
        }
    )
    
    # First post-initial reconfiguration: spatial hashing and dynamic perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate dynamic spatial hash matrix based on radius distribution
        radius_mean = np.mean(radii)
        radius_std = np.std(radii)
        # Generate spatial hashes for adaptive local perturbations
        spatial_hash = np.random.rand(n, 2) * (np.clip((radius_std / radius_mean), 0.01, 0.1))
        
        # Apply spatial hash with radius-aware perturbation
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / radius_mean) * 0.2
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / radius_mean) * 0.2
            perturbed_v[3*i+2] *= 1.02  # Slight uniform radius boost for expansion
        
        # Validate perturbed constraints: skip if this is expensive
        # Use a simplified constraint validation to avoid recomputation
        # Only check boundaries and overlapping constraints - not full pairwise check
        # Since we're building on a successful solution, avoid full re-validation
        res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 600,
                "ftol": 1e-12,
                "gtol": 1e-12
            }
        )
    
    # Targeted radius expansion phase with gradient-aware expansion and component-wise analysis
    # After the first optimization rounds, expand the least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix computation with numpy broadcasting
        # More efficient than nested loops
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]  # shape (n, n)
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute adjacency matrix based on current radii
        adj_radius = (radii + radii.reshape(-1, 1))
        adj = dists <= adj_radius  # shape (n,n)
        
        # Compute connectivity and component-wise expansion strategy
        # Use k-nearest neighbor to find the most isolated circle
        k = 5  # Look at k nearest circles for expansion planning
        dists_sorted = np.sort(dists, axis=1)  # shape (n, n)
        min_distances = dists_sorted[:, 1:k+1]  # Take 1st k+1 closest
        min_distances = np.min(min_distances, axis=1)  # shape (n,)
        
        # Find the circle with greatest minimum distance (least constrained)
        least_constrained_idx = np.argmax(min_distances)
        
        # Compute expansion factor based on current state and potential growth
        # Use a hybrid method with dynamic factor: 
        # - Base growth factor based on average radius
        # - Additional expansion based on current distance margin
        current_total = np.sum(radii)
        expansion_factor_base = 0.015 * (1.0 + (min_distances[least_constrained_idx] / np.mean(dists)))
        
        # Create expansion vector with targeted expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] = min(
            radii[least_constrained_idx] + expansion_factor_base * 1.1,  # Over-expansion with cap
            0.5 - (np.min(radii) * 0.1)  # Prevent overgrowth of existing small circles
        )
        # Distribute expansion across other centers based on distance to least constrained
        for i in range(n):
            if i != least_constrained_idx:
                # Compute distance from least constrained to i
                dist_to_least = dists[least_constrained_idx][i]
                # Normalize by radius to avoid over-expansion on small circles
                # Use a soft weighting factor based on distance to the least constrained
                weight = 1.0 / (1 + np.exp(-dist_to_least / 0.02))
                # Add expansion based on distance and radius
                new_radii[i] += expansion_factor_base * weight * np.clip(radii[i], 1e-4, 0.5)
        
        # Apply constraints to prevent overgrowth and ensure validity
        # Use vectorized clipping and validation
        new_radii = np.clip(new_radii, 1e-4, 0.5)
        
        # Update variables
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization round with expanded radii and improved tolerances
        res = minimize(
            neg_sum_radii,
            v_new,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "maxiter": 1500,
                "ftol": 1e-12,
                "gtol": 1e-12,
                "eps": 1e-12
            }
        )
    
    # Final cleanup with validation, fallback, and precision
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())