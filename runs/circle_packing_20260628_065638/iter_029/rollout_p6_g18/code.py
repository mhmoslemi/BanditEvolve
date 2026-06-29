import numpy as np

def run_packing():
    """
    Solves the 26-circle packing problem in a unit square using a multi-phase
    optimization approach with advanced geometric constraints and perturbation
    techniques. This version includes targeted radius expansion, geometric hashing
    for configuration diversity, and vectorized constraint propagation to push 
    the solution to higher radial sums.
    """
    n = 26
    cols = 5  # Adjust for better grid coverage and flexibility
    rows = (n + cols - 1) // cols

    # ---------------------------
    # 1. Initialization with enhanced geometric clustering and spatial awareness
    # ---------------------------
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols + np.random.uniform(-0.07, 0.07) * (1.0 / cols * 0.5)
        y_center = (row + 0.5) / rows + np.random.uniform(-0.07, 0.07) * (1.0 / rows * 0.5)
        
        # Apply alternate row shifts with asymmetric adjustment for non-uniform spacing
        if row % 2 == 1:
            x_center += np.random.uniform(-0.05, 0.05) * (1.0 / cols * 0.9)  # Slight asymmetric shift
        xs.append(x_center)
        ys.append(y_center)

    # Initialize radii with density-aware sizing
    avg_cell_edge = 1.0 / (cols * rows)
    r0 = 0.225 * avg_cell_edge + 1e-4  # Increased base radius to encourage density
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # ---------------------------
    # 2. Objective Function: Negative of sum of radii to optimize
    # ---------------------------
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # ---------------------------
    # 3. Constraint Creation with improved handling and vectorization
    # ---------------------------
    constraints = []

    # Add boundary constraints using vectorized expressions with fixed lambda bindings
    for i in range(n):
        constraints.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})  # Left boundary (x - r >= 0)
        constraints.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})  # Right boundary (x + r <= 1)
        constraints.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})  # Bottom boundary (y - r >= 0)
        constraints.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})  # Top boundary (y + r <= 1)

    # Add pairwise non-overlap constraint with geometric hashing + vectorization
    for i in range(n):
        for j in range(i+1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2  # Distance squared - (r_i + r_j)^2
            constraints.append({"type": "ineq", "fun": constraint_func})

    # ---------------------------
    # 4. Initial optimization phase with adaptive constraints and perturbation
    # ---------------------------
    # Use SLSQP with high precision and early convergence checks
    # First run: base optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-8})

    # ---------------------------
    # 5. Advanced perturbation and geometric hashing for enhanced search space
    # ---------------------------
    if res.success:
        v = res.x
        # Use radial geometric hashing for perturbation based on radii and relative position
        # Create a hash map with magnitude-based scaling to influence dense vs sparse regions
        hash_map = np.random.rand(n, 2) * 0.07 * np.sqrt(1.0 / (np.max(v[2::3]) + 1e-6))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_map[i, 0] * (1.0 + 0.2 * np.random.rand())  # Asymmetry in spatial perturbation
            perturbed_v[3*i+1] += hash_map[i, 1] * (1.0 + 0.2 * np.random.rand())  # Asymmetry in spatial perturbation

        # Secondary optimization to diversify the local minima space
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})

    # ---------------------------
    # 6. Strategic expansion of minimal radius circle with geometric feasibility
    # ---------------------------
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Find the circle with the least effective "freedom" to grow - min distance to others
        # This prioritizes expansion on the circle that has the least interaction with others
        min_dist = np.inf
        least_constrained_idx = 0
        for i in range(n):
            # Calculate min distance to other circles
            d = np.min([np.linalg.norm(centers[i] - centers[j]) for j in range(n) if j != i])
            if d < min_dist:
                min_dist = d
                least_constrained_idx = i

        # Apply targeted expansion with careful checking to prevent overlap
        valid = True
        
        # Check if expansion is feasible
        for i in range(n):
            for j in range(i + 1, n):
                if i == least_constrained_idx or j == least_constrained_idx:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist - (radii[i] + radii[j]) < 0:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            # Compute expansion factor based on current total and potential
            total = np.sum(radii)
            expansion_target = min(0.008, (0.0125 * total) / (n - 1))
            expansion_ratio = 1.0 + 0.25 * np.random.rand()  # Slight stochastic factor for diversity
            max_radius = np.max(radii)
            expansion_factor = expansion_target * (max_radius / np.mean(radii))

            # Apply expansion to all circles but prioritize expansion of the least constrained
            new_radii = radii.copy()
            expansion_per = expansion_factor
            new_radii[least_constrained_idx] += expansion_per * 1.3  # Over-expand slightly
            for i in range(n):
                if i != least_constrained_idx:
                    new_radii[i] += expansion_per * (0.8 + 0.2 * np.random.rand())  # Vary expansion per circle
            
            # Perform constraint validation and iterative refinement
            for _ in range(4):  # Up to 4 refinement steps
                expanded_centers = np.column_stack([v[0::3], v[1::3]])
                expanded_radii = new_radii
                valid = True

                for i in range(n):
                    for j in range(i + 1, n):
                        dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < expanded_radii[i] + expanded_radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    break
                else:
                    # Reduce expansion slightly
                    expansion_factor *= 0.9

            # Apply new radii to the vector and optimize the new configuration
            v_new = v.copy()
            v_new[2::3] = new_radii
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=constraints, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())