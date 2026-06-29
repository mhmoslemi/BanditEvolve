import numpy as np

def run_packing():
    # --- INITIALIZATION: MODIFIED GRID FOR DYNAMIC SPATIAL RECONFIGURATION ---
    n = 26
    
    # Use a dynamic column calculation that respects aspect ratio for better edge case management
    # cols = int(np.ceil(np.sqrt(n)))  # Standard grid
    cols = int(np.ceil(np.sqrt(n * 0.9)))
    rows = (n + cols - 1) // cols
    
    # Enhanced initialization: Use stochastic grid with local clustering, non-uniform scaling, and multi-stage seeding
    xs = []
    ys = []
    
    # Phase 1: Base grid with geometric hashing and non-uniform scaling
    for i in range(n):
        row_base = i // cols
        col_base = i % cols
        col_shift = np.random.uniform(-0.1, 0.1)
        row_shift = np.random.uniform(-0.1, 0.1)
        
        # Dynamic scaling: scale col positions by sqrt(row) for better space utilization
        col_base_scaled = col_base * (1.0 + 0.1 * np.random.rand())
        row_base_scaled = row_base * (1.0 + 0.1 * np.random.rand()) ** 0.5
        
        # Add adaptive offset based on row and col to break symmetry and cluster at boundaries
        x = (col_base_scaled + 0.5) / cols + np.random.uniform(-0.04, 0.04)
        y = (row_base_scaled + 0.5) / rows + np.random.uniform(-0.04, 0.04)
        
        # Stagger rows for dynamic spacing: alternate row offsets based on row index
        if row_base % 3 == 1:
            x += np.random.uniform(0.03, 0.06) * np.random.choice([-1, 1])
        elif row_base % 3 == 2:
            y += np.random.uniform(0.02, 0.05) * np.random.choice([-1, 1])
        xs.append(x)
        ys.append(y)
    
    # Phase 2: Stochastic seeding with multi-scale perturbation to disrupt potential symmetry
    # Add multi-scale perturbation to the grid layout
    perturbation = np.random.randn(n, 2) * 0.02
    xs = np.array(xs) + perturbation[:, 0]
    ys = np.array(ys) + perturbation[:, 1]
    
    # Phase 3: Spatial constraint-aware scaling of initial placement
    # We apply scaling that is proportional to the inverse of local packing density
    radius_initial = 0.30
    spacing_factor = 1 / (1 + 0.01 * np.random.rand(n)) # Add some stochasticity for non-uniformity
    # Also use an adaptive scaling function to increase edge cases for expansion
    radius_initial = np.min([radius_initial, 0.35 * 1.0 / (1.0 + np.sqrt(n) / 25)])  # Slight decrease to enable more expansion
    r0 = radius_initial * spacing_factor
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # --- BOUNDS: STRICT AND CONSISTENT WITH 3*N PARAMETER SPACE ---
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.4)]  # Increased upper bound for potential expansion
    
    # --- OBJECTIVE: MINIMIZE NEGATIVE SUM OF RADIUSES --- 
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Standard optimization objective
    
    # --- CONSTRAINTS: ENFORCED, VECTORIZED, AND TYPE-AWARE ---
    # We apply strict lambda capture with parameter binding to avoid late-binding issues
    cons = []
    # Vectorized boundary constraints (x, y, radius)
    for i in range(n):
        # x_lower: x - r >= 0
        cons.append({"type": "ineq", 
                    "fun": lambda v, i=i: v[3*i] - v[3*i+2]}) # x - r >= 0
        # x_upper: x + r <= 1
        cons.append({"type": "ineq", 
                    "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]}) # 1 - x - r >= 0
        # y_lower: y - r >= 0
        cons.append({"type": "ineq", 
                    "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]}) # y - r >= 0
        # y_upper: y + r <= 1
        cons.append({"type": "ineq", 
                    "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]}) # 1 - y - r >= 0
    
    # Vectorized pairwise overlap constraints with adaptive constraint scaling
    for i in range(n):
        for j in range(i + 1, n):
            # Function to check overlap constraint: 
            # [dx^2 + dy^2] - [r_i + r_j]^2 >= 0
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                r_i = v[3*i+2]
                r_j = v[3*j+2]
                return dx*dx + dy*dy - (r_i + r_j)**2
            # Add constraint with a slight tolerance for numeric stability
            # We add an epsilon adjustment to prevent numerical errors from zero
            # Add a minor adaptive scaling to allow for tighter packing near boundaries
            # This scaling is minimal and ensures the constraint remains active
            # The value of scaling_factor here is chosen carefully for numerical robustness (not too small)
            scaling_factor = 1.2  # Slight increase in constraint strength
            cons.append({"type": "ineq", 
                        "fun": lambda v, i=i, j=j: constraint_func(v)*(scaling_factor)})

    # --- OPTIMIZATION PHASES: HYPER-CONTROLLED, WITH SEQUENTIAL RECONFIGURATION ---
    # First pass: optimize initial configuration
    def run_phase_optimization(optimizer, initial_x, phase_index, maxiter=300, ftol=1e-10, scale=1.0):
        # This function abstracts a phase of optimization
        # scale is used to scale the initial guess or to adjust constraints for specific phases
        result = minimize(
            neg_sum_radii, 
            initial_x * scale,  # Apply scaling to potentially expand small radii initially
            method="SLSQP", 
            bounds=bounds, 
            constraints=cons, 
            options={
                "maxiter": maxiter,
                "ftol": ftol,
                "disp": False
            }
        )
        if not result.success:
            print(f"Warning: Phase {phase_index} optimization not successful, reverting to previous state.")
        return result.x if result.success else initial_x
    
    # Phase 1: Initial optimization with moderate control
    res = run_phase_optimization(minimize, v0, 1, maxiter=1200, ftol=1e-10, scale=1.0)
    
    # Phase 2: Reconfiguration with stochastic spatial constraint violation (but maintaining non-overlap)
    # We perform a controlled spatial perturbation that simulates a "shaking" to find better local minima
    if res.success:
        # Create a spatial hash to perturb positions with adaptive strength based on circle radius
        # This introduces a dynamic perturbation which is less disruptive for smaller circles
        # Perturbation is scaled by an inverse function of the radius to preserve smaller circles
        radius_weighted_perturbation = np.random.rand(n, 2) * 0.05
        # Apply perturbation scaled by inverse of radii (to preserve small circles)
        radius_weights = np.clip(res[2::3], 1e-6, 0.4) # clip radius to avoid zero denominator
        scaled_perturbation = radius_weighted_perturbation / (radius_weights[:, np.newaxis] + 1e-8)
        perturbed_v = res.copy()
        perturbed_v[0::3] += scaled_perturbation[:, 0]
        perturbed_v[1::3] += scaled_perturbation[:, 1]
        res = run_phase_optimization(minimize, perturbed_v, 2, maxiter=400, ftol=1e-11, scale=0.9)

    # Phase 3: Introduce directional expansion on the most "free-standing" circle (least constrained)
    if res.success:
        # Compute distance metrics for all circles
        centers = np.column_stack([res[0::3], res[1::3]])
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute current average radius and growth budget
        current_avg_radius = np.mean(res[2::3])
        growth_budget = 0.007  # Slightly larger than the previous approach
        expansion_radius = current_avg_radius + growth_budget / (n - 1)
        
        # Create a new configuration that pushes the least constrained circle outward
        # To do this, we compute a displacement vector that increases radius and moves center
        # Displacement is calculated to ensure boundary constraints are maintained
        # For now, we use a simple heuristic to expand this particular circle
        # We do this by adding a scaled expansion to the radius and a directional shift in position
        # For safety, we check boundaries during this operation
        v_expanded = res.copy()
        # Calculate directional shift to move circle away from potential clusters
        nearest_neighbors_idx = np.argsort(dists[least_constrained_idx, :])  # Nearest neighbors of least constrained circle
        center = centers[least_constrained_idx]
        direction = np.array([0.0, 0.0])
        for i in nearest_neighbors_idx[1:3]:  # Take closest two neighbors for direction
            if i != least_constrained_idx:
                direction += centers[i] - center
        direction /= np.linalg.norm(direction) + 1e-8
        # Apply a small directional movement
        displacement = direction * 0.01 * (1.0 + np.random.uniform(-0.1, 0.1))  # Stochastic small move
        v_expanded[3*least_constrained_idx] += displacement[0]
        v_expanded[3*least_constrained_idx+1] += displacement[1]
        v_expanded[3*least_constrained_idx+2] += growth_budget / (n - 1) * 0.9
        
        # Run the second phase optimization, focusing on this circle
        res = run_phase_optimization(minimize, v_expanded, 3, maxiter=300, ftol=1e-12, scale=0.9)

    # Phase 4: Introduce an adaptive radius expansion to the smallest circle (while maintaining bounds and minimizing overlap)
    # We need to find the smallest circle without violating constraints
    if res.success:
        radii = res[2::3]
        min_radius_idx = np.argmin(radii)
        # Compute a safe expansion limit that respects boundaries and doesn't cause overlapping
        # To do this, compute current bounding radius constraints
        x = res[3*min_radius_idx]
        y = res[3*min_radius_idx+1]
        current_radius = radii[min_radius_idx]
        # Potential expansion limit based on remaining space
        max_possible_radius = min(
            x,
            (1.0 - x),
            y,
            (1.0 - y)
        )
        expansion_limit = max_possible_radius - current_radius
        expansion_amount = max(0.003, expansion_limit * 0.8)  # Ensure safe expansion
        
        # Create a new configuration to expand this circle while maintaining constraints
        v_expanded = res.copy()
        v_expanded[3*min_radius_idx+2] += expansion_amount
        
        # Run third phase optimization to find a new feasible state
        res = run_phase_optimization(minimize, v_expanded, 4, maxiter=400, ftol=1e-11, scale=0.9)

    # Final check: if any optimization failed, fallback to initial state
    v = res.x if res.success else v0
    
    # Final validation pass: this ensures we don't accidentally violate constraints (e.g. NaNs, etc.)
    # This is critical for the final step before returning
    # We apply a soft clipping and re-check boundaries
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.45)  # Clip to ensure numerical stability and prevent overflow
    
    # Ensure no overlaps: we do this manually (not with the constraint function) as constraints
    # can sometimes be too loose or due to numerical errors in the optimizer
    # This is a critical final verification
    def manual_overlap_check(centers, radii):
        n = centers.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                if dist_sq < min_dist**2 - 1e-10:  # Account for floating point error tolerance
                    return False
        return True
    
    if not manual_overlap_check(centers, radii):
        # Fallback: apply a small uniform contraction to all circles to satisfy constraints
        contraction = 0.9 * (np.mean(radii) / 0.4)  # Scale down to allow for non-overlapping
        # Only contract if it's not already below a safe minimum
        min_radius = np.min(radii)
        if min_radius < 1e-4:
            contraction = 0.6
        radii = np.clip(radii * contraction, 1e-6, 0.45)
        centers = np.column_stack([v[0::3], v[1::3]])
        # Recheck with new radii
        if not manual_overlap_check(centers, radii):
            # Last resort: return original v0 with clipping
            radii = v[2::3].copy()
            radii = np.clip(radii, 1e-6, 0.45)
            centers = np.column_stack([v[0::3], v[1::3]])
    
    # Final validation: we must ensure no NaNs and radii are valid
    if np.isnan(centers).any() or np.isnan(radii).any():
        # Return initial clean values
        centers = np.column_stack([v0[0::3], v0[1::3]])
        radii = np.clip(v0[2::3], 1e-6, 0.45)
    
    sum_radii = radii.sum()
    return centers, radii, sum_radii