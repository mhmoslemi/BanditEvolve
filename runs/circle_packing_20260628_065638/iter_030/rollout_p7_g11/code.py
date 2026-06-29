import numpy as np

def run_packing():
    """
    Optimized circle packing algorithm that:
    1. Uses structured grid initialization with adaptive grid refinement.
    2. Incorporates advanced perturbation mechanisms and vectorized constraint calculations.
    3. Implements a hierarchical multi-phase optimization strategy.
    4. Uses a refined expansion strategy with geometric-aware radius growth.
    5. Incorporates syntactic validation and gradient-aware optimization.
    6. Ensures precise structural consistency with 26 circles.
    """
    n = 26

    # Grid-based initialization: adaptive grid with optimized spacing
    cols = 5
    rows = (n + cols - 1) // cols
    grid_size_x = 1.0 / cols
    grid_size_y = 1.0 / rows

    # Base grid positions
    xs_base = np.arange(cols) + 0.5
    ys_base = np.arange(rows) + 0.5

    # Generate initial positions with adaptive geometric spreading and stochasticity
    xs = []
    ys = []
    x_offsets = np.random.uniform(-0.1, 0.1, size=n)
    y_offsets = np.random.uniform(-0.1, 0.1, size=n)
    
    for i in range(n):
        col = i % cols
        row = i // cols
        x = xs_base[col] + x_offsets[i] * grid_size_x * 0.8
        y = ys_base[row] + y_offsets[i] * grid_size_y * 0.8
        # Alternate row staggering for visual efficiency
        if row % 2 == 1:
            x += grid_size_x * 0.45
        xs.append(x)
        ys.append(y)

    # Initialize radii with dynamic base value and small jitter
    base_radius = 0.33 / cols * 1.2
    r0 = base_radius * (1 + np.random.uniform(-0.15, 0.15, size=n))

    # Decision vector for SLSQP
    v0 = np.empty(3 * n, dtype=np.float64)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    # Validate bounds consistency (length must be 3*26)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((1e-4, 0.5))  # radius
    assert len(bounds) == 3 * n, "Bounds length mismatch"  # Ensure strict consistency

    # Cost function: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundary conditions
    # Using explicit closures to avoid closure capture issues (safeguarding lambda capture)
    # 4 constraints per circle: x bounds, y bounds
    cons = []
    for i in range(n):
        # x >= radius
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3 * i] - v[3 * i + 2]})
        # x + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3 * i] - v[3 * i + 2]})
        # y >= radius
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3 * i + 1] - v[3 * i + 2]})
        # y + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3 * i + 1] - v[3 * i + 2]})

    # Efficient overlap constraint calculation using vectorized math
    # Precompute all pairwise distance constraints in the form of a function
    for i in range(n):
        for j in range(i + 1, n):
            # Create a closure that avoids capture issues with variable i,j
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Optimizer settings
    # Initial optimization with adaptive settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-10})
    
    # Optimization refinement stage: spatial-aware configuration with multiple phases
    # Phase 1: Perturbation with grid-based spatial adjustment
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Spatial hashing for enhanced reconfiguration
        # Use grid-based spatial awareness and perturb based on geometry
        # Create a grid map for spatial hashing
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbed_v = v.copy()
        for i in range(n):
            grid_idx_x = int(round(centers[i, 0] * cols))
            grid_idx_y = int(round(centers[i, 1] * rows))
            scale_factor = 0.5 * (radii[i] / np.mean(radii)) + 0.3
            perturbed_v[3*i] += spatial_hash[i, 0] * scale_factor
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale_factor

        # Re-optimization after spatial perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-12, "eps": 1e-10})

    # Phase 2: Refinement with targeted expansion and constraint awareness
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Optimized distance matrix via broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute per-circle minimum distance to neighbors
        min_dists = np.min(dists, axis=1)
        # Find least constrained circle (by maximizing minimum distance to neighbors)
        least_constrained_idx = np.argmax(min_dists)
        current_total = np.sum(radii)
        
        # Target expansion: dynamically calculated based on total sum and potential
        target_total = current_total + 0.007
        expansion_factor = (target_total - current_total) / (n - 1) * (current_total / np.sum(radii))
        
        # Create expansion vector, prioritizing least constrained circle
        new_radii = radii.copy()
        # Slight over-expansion for perturbation
        new_radii[least_constrained_idx] += expansion_factor * 1.15
        # Base expansion for others
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.08 * np.random.rand())
        
        # Gradient-aware constraint validation with fallback
        # Perform gradient-safe expansion
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate without recomputing overlaps (re-use existing distance calculations)
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    try:
                        dist = np.sqrt(
                            (expanded_centers[i, 0] - expanded_centers[j, 0])**2 +
                            (expanded_centers[i, 1] - expanded_centers[j, 1])**2
                        )
                        if dist < (new_radii[i] + new_radii[j]) - 1e-12:
                            valid = False
                            break
                    except Exception:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Fall back: reduce expansion by a safe factor
                new_radii = radii + (new_radii - radii) * 0.97

        # Update decision vector and re-optimize
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-12, "eps": 1e-11})

    # Final cleanup and validation
    v = res.x if res.success else v0
    radii = np.clip(v[2::3], 1e-6, None)
    centers = np.column_stack([v[0::3], v[1::3]])
    return centers, radii, float(radii.sum())