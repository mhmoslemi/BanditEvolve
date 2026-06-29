import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Use grid-aligned but randomized initialization to balance symmetry and 
    # allow for local deformations through constraint-based optimization
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce localized randomness for asymmetric perturbation potential
        x = x_center + np.random.uniform(-0.07, 0.07)
        y = y_center + np.random.uniform(-0.07, 0.07)
        # Introduce staggered offset with row-dependent magnitude
        if row % 2 == 1:
            x += np.random.uniform(-0.03, 0.03) * (0.5 / cols)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.32 / cols - 1e-3  # Tuned from 0.35 to allow for larger total sum potential
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Ensure bounds list length matches 3*n and is exactly aligned with 3*26 positions
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # All circles have radii >=1e-4
        
    # Objective function to maximize sum of radii (minimize negative)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraint definitions using lambda closures with bounded i capture to prevent
    # closure capture issues in nested functions
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
    
    # Overlap constraints between any pair of circles
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda closure with capture to ensure correct i,j binding
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})
    
    # Initial optimization with increased max iterations and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1600, "ftol": 1e-11, "eps": 1e-10})
    
    # Asymmetric reconfiguration: stochastic perturbation of least constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Find least constrained circle (max minimum distance to all others)
        distances = np.zeros(n)
        for i in range(n):
            min_dist = np.inf
            for j in range(n):
                if i == j:
                    continue
                dist = np.hypot(centers[i, 0] - centers[j, 0], 
                               centers[i, 1] - centers[j, 1])
                if dist < min_dist:
                    min_dist = dist
            distances[i] = min_dist

        least_constrained_idx = np.argmax(distances)
        
        # Introduce stochastic expansion to least constrained circle with 
        # directional spatial constraints to unlock new packing potential
        # Add directional spatial noise to allow for asymmetric rearrangement
        perturbation_factor = 0.03  # Tuned to avoid excessive deviation
        perturbation = np.random.rand(2) * perturbation_factor
        v[3*least_constrained_idx] += perturbation[0]
        v[3*least_constrained_idx+1] += perturbation[1]
        
        # Re-evaluate with perturbed configuration and re-optimized
        v_perturbed = v.copy()
        res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10})
    
    # Targeted radius expansion on least constrained circle with 
    # dynamic expansion rate based on current configuration and potential for growth
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Re-calculate distances to ensure accurate evaluation
        distances = np.zeros(n)
        for i in range(n):
            min_dist = np.inf
            for j in range(n):
                if i == j:
                    continue
                dist = np.hypot(centers[i, 0] - centers[j, 0], 
                               centers[i, 1] - centers[j, 1])
                if dist < min_dist:
                    min_dist = dist
            distances[i] = min_dist
        least_constrained_idx = np.argmax(distances)
        
        # Calculate growth based on available spatial budget
        # Current sum + potential for expansion without overlap
        current_sum = np.sum(radii)
        max_possible_growth = 0
        
        # Estimate max possible expansion for least constrained circle
        for j in range(n):
            if j == least_constrained_idx:
                continue
            # Estimate available space: distance - current radii
            dx = centers[least_constrained_idx, 0] - centers[j, 0]
            dy = centers[least_constrained_idx, 1] - centers[j, 1]
            dist = np.hypot(dx, dy)
            possible_growth = dist - (radii[least_constrained_idx] + radii[j]) 
            if possible_growth > 0:
                max_possible_growth += possible_growth
        
        max_possible_growth = max(0.005, max_possible_growth)  # Minimal safety padding
        # Apply expansion with proportional distribution
        expansion = max_possible_growth * (1.1)  # 10% buffer for robustness
        
        # Apply expansion to all except least constrained
        for i in range(n):
            if i == least_constrained_idx:
                continue
            # Compute proportional expansion based on available space
            dx = centers[i, 0] - centers[least_constrained_idx, 0]
            dy = centers[i, 1] - centers[least_constrained_idx, 1]
            dist = np.hypot(dx, dy)
            # Calculate expansion rate based on distance to least constrained
            # Expand more if closer to less constrained circles
            expansion_rate = 1.0 + 0.5 * (dist - (radii[i] + radii[least_constrained_idx])) / dist
            v[3*i + 2] += expansion * expansion_rate
        
        # Re-evaluate with expanded configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10})
    
    # Final optimization for robustness and precision
    if res.success:
        v = res.x
        radii = v[2::3]
        # Revalidate all pairwise distances to ensure no overlap
        centers = np.column_stack([v[0::3], v[1::3]])
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.hypot(dx, dy)
                if dist < (radii[i] + radii[j]) - 1e-12:
                    # If overlap detected, slightly decrease radii to preserve validity
                    if radii[i] > 1e-4:
                        radii[i] -= (radii[i] + radii[j] - dist + 1e-12) * 0.5
                    if radii[j] > 1e-4:
                        radii[j] -= (radii[i] + radii[j] - dist + 1e-12) * 0.5
        
        # Clip radii to prevent unbounded growth
        radii = np.clip(radii, 1e-6, 0.5)
        # Update vector with clipped radii
        v[2::3] = radii
        
        # Final optimization with fixed configuration to ensure precision
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-10})
    
    # Final validation
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())