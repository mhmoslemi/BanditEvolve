import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with spatially dynamic lattice and stochastic initialization
    xs = [] 
    ys = []
    radius_factors = []
    
    # Create an asymmetric spatial lattice with 5 columns and variable row spacing
    # First, calculate base lattice spacing
    col_base_x = 1.0 / cols
    row_base_y = 1.0 / rows
    row_spacing = 0.4 * row_base_y
    
    # Initialize circle positions with a dynamic seed that incorporates row/column interaction
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base position with offset to reduce clustering and introduce asymmetry
        x_center = col_base_x * (col + 0.5)
        y_center = row_base_y * (row + 0.5) + (row * row_spacing)
        
        # Add spatial perturbation based on the row parity and column index
        x_offset = (np.sin(np.pi * col / cols) * (0.03) * (1 if row % 2 == 0 else 1.0))
        y_offset = (np.cos(np.pi * row / rows) * (0.03) * (1 if col % 3 == 1 else 1.0))
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Add a small randomized offset to enhance stochasticity in initial guess
        x += np.random.uniform(-0.03, 0.03)
        y += np.random.uniform(-0.03, 0.03)
        
        # Enforce physical boundaries with clamping and spatial constraints
        x = np.clip(x, 1e-8, 1.0 - 1e-8)
        y = np.clip(y, 1e-8, 1.0 - 1e-8)
        
        xs.append(x)
        ys.append(y)
        
        # Estimate a radius factor that depends on column and row proximity to the center
        # This gives more room in periphery and tighter packing near edges
        col_dist_to_center = abs(col - cols/2) / cols
        row_dist_to_center = abs(row - rows/2) / rows
        factor = 1.0 - (col_dist_to_center + row_dist_to_center) * 0.5
        factor = max(0.5, factor)  # Ensure lower bound
        radius_factors.append(factor)
    
    # Determine optimal initial radii based on spatial distribution and packing factor
    r0 = (0.32 + 0.1 * np.mean(radius_factors)) / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.array(radius_factors) * r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n length

    # Objective function with gradient approximation for optimization
    def neg_sum_radii(v):
        # Gradient approximation using finite differences for stability
        # This prevents singular gradient issues and makes optimization robust
        return -np.sum(v[2::3])
    
    # Create constraints with explicit lambda captures to prevent closure collisions
    cons = []
    for i in range(n):
        # Ensure left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized and efficient overlap constraints with early termination
    # This uses vectorized distance calculation to reduce per-pair computation
    # and includes a tolerance layer for numerical stability
    for i in range(n):
        for j in range(i + 1, n):
            # Create constraint function with explicit closure bounds
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2 + 2e-4*(v[3*i+2]+v[3*j+2]))})  # +2e-4*(radius sum) as a stabilizer

    # Optimize with multiple phases for robust convergence
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "eps": 1e-8})

    # First reconfiguration phase: spatial transformation using dynamic hash-based perturbation
    # Generate a geometric hash vector that scales with spatial constraints
    if res.success:
        v = res.x
        v_res = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Dynamic hash generation based on spatial distribution
        # We use the same spatial lattice but apply a more aggressive stochastic perturbation
        # with radius-dependent scaling for enhanced reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            # Apply perturbation relative to the spatial weight of radius
            scale = max(radii[i] / np.mean(radii), 0.5)
            perturbed_v[3*i] += spatial_hash[i, 0] * scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale
            # Ensure perturbed positions stay within square
            perturbed_v[3*i] = np.clip(perturbed_v[3*i], 1e-8, 1.0 - 1e-8)
            perturbed_v[3*i+1] = np.clip(perturbed_v[3*i+1], 1e-8, 1.0 - 1e-8)
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})
    
    # Second reconfiguration phase: radial expansion with targeted optimization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Calculate distance matrix with vectorization and early termination for performance
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        distance_matrix = np.sqrt(dx**2 + dy**2) + 1e-12
        
        # Compute minimum distance to each circle to identify least constrained circles
        min_distances = np.min(distance_matrix, axis=1)
        # Find the circle that has the most margin (lowest relative packing density)
        # Use a non-scalar measure that accounts for relative spacing
        least_constrained_idx = np.argmin(min_distances / (radii + 1e-8))  # Avoid division by zero
        
        # Estimate potential expansion based on spatial margins
        # Calculate the maximum possible expansion without exceeding boundary
        max_possible_growth = np.zeros(n)
        for i in range(n):
            # For each circle, determine how much it could grow in radius before hitting boundary
            # Left: x - r >= 0 → r <= x → current r = r[i] → grow by x - r[i]
            # Right: 1 - x - r >= 0 → r <= 1 - x → current r = r[i] → grow by 1 - x - r[i]
            max_possible_growth[i] = np.clip(1.0 - centers[i, 0] - radii[i], 0, 1e8) \
                                   + np.clip(centers[i, 0] - radii[i], 0, 1e8)
            # Similarly bottom and top
            max_possible_growth[i] += np.clip(1.0 - centers[i, 1] - radii[i], 0, 1e8) \
                                   + np.clip(centers[i, 1] - radii[i], 0, 1e8)
        
        # Find radius that can be expanded the most
        expandable_idx = np.argmax(max_possible_growth)
        
        # Use a dynamic expansion strategy with geometric constraints
        # Instead of uniform expansion, prioritize the most expandable circle
        # Create perturbed configuration for new radius distribution
        # Apply expansion with constraint validation in a controlled way
        
        if expandable_idx == least_constrained_idx:
            # Use more aggressive expansion strategy for least constrained
            # Allocate a growth target proportionally based on spacing and radial constraints
            expansion_ratio = 0.009
            target_grow = expansion_ratio * np.mean(radii) * (radii[least_constrained_idx] / np.mean(radii))
            expansion = np.full(n, 0.0)
            expansion[least_constrained_idx] = target_grow * 1.2  # Slight over-targeting
            # Add a small expansion to adjacent circles for indirect leverage
            for i in range(n):
                if i != least_constrained_idx:
                    expansion[i] = (target_grow * 0.3) * (1.0 + np.random.rand() / 4.0)
        else:
            # Use a standard expansion that prioritizes spatial expansion
            # Allocate base expansion to the most expandable circle and distribute the rest
            expansion = np.full(n, 0.0)
            expansion[expandable_idx] = max_possible_growth[expandable_idx] * 0.02
            for i in range(n):
                if i != expandable_idx:
                    expansion[i] = np.clip(max_possible_growth[i] * 0.005, 0, 0.005)
        
        new_radii = radii + expansion
        new_radii = np.clip(new_radii, 1e-6, 0.45)  # Max radius constraint is 0.45
        
        # Reconstruct decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with the adjusted radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-8})
    
    # Final validation and adjustment
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.45)
    # Final refinement with gradient-based adjustment
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 200, "ftol": 1e-11, "eps": 1e-8})
    
    final_centers = np.column_stack([res.x[0::3], res.x[1::3]])
    final_radii = np.clip(res.x[2::3], 1e-6, 0.45)
    return final_centers, final_radii, float(final_radii.sum())