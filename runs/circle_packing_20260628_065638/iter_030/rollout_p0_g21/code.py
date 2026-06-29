import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    # Use advanced grid-based grid + adaptive perturbation initialization with 
    # multi-stage radius adjustment
    
    # Initialize via a hybrid approach: geometric grid + randomized perturbation + 
    # adaptive spacing for better early-stage separation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        
        # Base grid spacing: use a grid with dynamic spacing 
        # Add randomized offsets with increasing variance as grid spacing is 
        # tighter for central areas
        offset_x = np.random.rand() * (0.15 - 0.02) + 0.02
        offset_y = np.random.rand() * (0.15 - 0.02) + 0.02
        
        # Add asymmetric shift for alternating rows as before
        asymmetric_shift = (0.5 / cols) if row % 2 == 1 else 0.0
        base_x += asymmetric_shift
        
        # Perturb based on base spacing: reduce jitter for tighter clusters 
        perturbation_ratio = 0.1 - 0.1*(row + col)**0.5 / (rows + cols)
        x = base_x + np.random.rand() * (offset_x * perturbation_ratio)
        y = base_y + np.random.rand() * (offset_y * perturbation_ratio)
        
        # Add a subtle spatial gradient to avoid grid alignment for better
        # distribution
        grad_x = (col + 1) / cols * 0.03
        grad_y = (row + 1) / rows * 0.03
        x += np.random.rand() * grad_x - grad_x / 2
        y += np.random.rand() * grad_y - grad_y / 2
        
        xs.append(x)
        ys.append(y)
    
    # Base radius: calculated via spatial awareness to avoid overinitialization
    # Base area per circle: calculate using row/col spacing with adaptive safety factor
    grid_width = cols * (1.0 / cols)  # Unit width 
    grid_height = rows * (1.0 / rows)  # Unit height 
    base_radius = (grid_width / np.sqrt(n)) * 0.85  # Safe margin to avoid clustering
    r0 = base_radius - 0.002  # 2% buffer due to perturbation
    
    # Initialize vector with dynamic spacing awareness
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Construct bounds with tight constraints and minimal ranges
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # same as before but more robust
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Create constraints with lambda closures but use positional parameters explicitly to avoid binding issues
    # Boundary constraints for all circles
    cons = []
    for i in range(n):
        # Left boundary: x[i] - r[i] >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary: x[i] + r[i] <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary: y[i] - r[i] >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary: y[i] + r[i] <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with efficient functional design and closure-aware implementation
    for i in range(n):
        for j in range(i + 1, n):
            # Use a closure-friendly approach with fixed i,j to avoid lambda binding problems
            def constraint_func_capturer(i, j):
                def constraint_func(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                return constraint_func
            cons.append({"type": "ineq", "fun": constraint_func_capturer(i, j)()})
    
    # First optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-12, "eps": 1e-12})
    
    # Safety validation layer
    if res.success:
        # Create local state copy with validation
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        
        # Check for NaNs and negative values immediately
        if np.isnan(radii).any():
            v = v0
            res = minimize(neg_sum_radii, v, method="SLSQP", 
                           bounds=bounds, constraints=cons, 
                           options={"maxiter": 300, "ftol": 1e-10})
        
        # Ensure radii are non-negative
        if (radii < 0).any():
            radii = np.clip(radii, 0, 1)
            v = np.concatenate([v[0::3], v[1::3], radii])
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # If not success, fall back with safety checks
    if not res.success:
        # Fallback: initialize with a more robust starting configuration
        # Perturb base grid with adaptive perturbation
        xs_restart = []
        ys_restart = []
        for i in range(n):
            row = i // cols
            col = i % cols
            base_x = (col + 0.5) / cols
            base_y = (row + 0.5) / rows
            # Perturbation with exponential decay to corners
            pert_x = np.random.normal(0, 0.05 * (1 / (row + 1) + 1/(col + 1)))
            pert_y = np.random.normal(0, 0.05 * (1 / (row + 1) + 1/(col + 1)))
            x = base_x + pert_x
            y = base_y + pert_y
            if row % 2 == 1:
                x += 0.5 / cols  # alternate row shift
            xs_restart.append(x)
            ys_restart.append(y)
        # Re-initialize radii with adaptive spacing
        r_restart = (0.25 / cols) * 1.2 - 0.001  # higher initial radii than base
        v_restart = np.array(xs_restart + ys_restart + r_restart.tolist())
        res = minimize(neg_sum_radii, v_restart, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 700, "ftol": 1e-11})
    
    # Second wave optimization after initial phase with gradient enhancements
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        # Generate a "constraint-aware" perturbation matrix using radius-aware scaling
        # Scale perturbation with radius to allow high-radius areas to expand 
        # while constraining low ones
        scale_factor = (v[2::3] / np.max(v[2::3])) * 0.8  # radius-aware scaling
        # Create a more structured perturbation matrix
        perturb_matrix = np.random.randn(n, 2) * (scale_factor + 0.05)
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturb_matrix[i, 0]
            perturbed_v[3*i+1] += perturb_matrix[i, 1]
        
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})
        v = res.x
    
    # Third stage: advanced optimization with gradient-enhanced and multi-scale constraints
    # Implement a "constraint priority" strategy: 
    # 1. First optimize by constraint type (boundary then overlap) 
    # 2. Then optimize by circle density 
    # 3. Finally optimize spatial distribution
    
    # Optimize by constraint type
    def constraint_priority_objective(v):
        weights = np.array([1.0, 1.0, 1.0, 1.0])  # equal weight
        # Compute constraint violations
        # This is a heuristic and may not be precise, but can help guide optimization
        v_center = v.reshape(n, 3)
        violations = []
        for i in range(n):
            # Check boundary constraints
            if v_center[i, 0] - v_center[i, 2] < 0:
                violations.append(0.1 * (0 - (v_center[i, 0] - v_center[i, 2])))
            if v_center[i, 0] + v_center[i, 2] > 1:
                violations.append(0.1 * (v_center[i, 0] + v_center[i, 2] - 1))
            if v_center[i, 1] - v_center[i, 2] < 0:
                violations.append(0.1 * (0 - (v_center[i, 1] - v_center[i, 2])))
            if v_center[i, 1] + v_center[i, 2] > 1:
                violations.append(0.1 * (v_center[i, 1] + v_center[i, 2] - 1))
            # Check overlap constraints (approximate violation)
            for j in range(i+1, n):
                dx = v_center[i, 0] - v_center[j, 0]
                dy = v_center[i, 1] - v_center[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < (v_center[i, 2] + v_center[j, 2]) - 1e-12:
                    violations.append(0.2 * ( (v_center[i, 2] + v_center[j, 2]) - dist ))
        return -np.sum(violations) + -np.sum(v[2::3])  # prioritize minimizing violations and then sum
    
    if res.success:
        res = minimize(constraint_priority_objective, v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 500, "ftol": 1e-9, "eps": 1e-12})
        v = res.x
    
    # Final optimization: implement a "gradient-enhanced" strategy
    # Using a hybrid of SLSQP and Nelder-Mead with adaptive constraints
    
    # Implement gradient-enhanced optimization by using a two-step process:
    # 1. Use SLSQP with more robust gradient handling
    # 2. Then use Nelder-Mead to escape local optima
    if res.success:
        res = minimize(neg_sum_radii, v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 400, "ftol": 1e-10, "eps": 1e-12})
        v = res.x
        res = minimize(neg_sum_radii, v, method="Nelder-Mead", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 200, "ftol": 1e-9})
        v = res.x
    
    # Final validation and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    # Additional constraint check for safety, even without optimize
    # This is a fast-check to ensure no out-of-bounds
    if not res.success:
        # Fallback validation
        for i in range(n):
            x, y, r = centers[i], centers[i,1], radii[i]
            if (x - r < -1e-12 or x + r > 1 + 1e-12 or 
                y - r < -1e-12 or y + r > 1 + 1e-12):
                # Fallback to safe configuration
                safe_centers = np.zeros((n,2))
                safe_radii = np.empty(n)
                for i in range(n):
                    safe_centers[i] = np.random.uniform(0.2, 0.8, 2)
                    safe_radii[i] = 0.03
                centers = safe_centers
                radii = safe_radii
                break
    return centers, radii, float(radii.sum())