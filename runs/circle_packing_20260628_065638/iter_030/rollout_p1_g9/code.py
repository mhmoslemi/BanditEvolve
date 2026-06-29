import numpy as np

def run_packing():
    n = 26
    cols = 5  # For 26 circles, a 6x5 grid gives better density than 5x5
    num_rows = (n + cols - 1) // cols  # Ensures grid covers all 26
    # Initialize via a hybrid of geometric hashing and adaptive jitter for better initial distribution
    
    # Generate base positions with grid spacing and adaptive geometric hashing
    xs_base = []
    ys_base = []
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        x_center = (col_idx + 0.75) / cols
        y_center = (row_idx + 0.75) / num_rows
        # Add jitter with non-uniform random for more distributional diversity and avoid symmetry
        x = x_center + np.random.uniform(-0.04, 0.04) * (1.0 - row_idx / num_rows)
        y = y_center + np.random.uniform(-0.04, 0.04) * (1.0 - row_idx / num_rows)
        # Alternate row row offset with adaptive spacing to simulate hexagonal packing in a rectangle
        if row_idx % 2 == 1:
            # Introduce offset that depends on the row's horizontal position for natural asymmetry
            x += (0.5 / cols) * (1 - (col_idx + 0.5)/cols) * np.random.uniform(-0.1, 0.1)
        xs_base.append(x)
        ys_base.append(y)
    
    r0 = 0.35 / cols - 1e-4  # Slightly smaller than parent's to allow for later expansion
    # Initialize v0 with the jittered positions
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs_base)
    v0[1::3] = np.array(ys_base)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Define the objective as negative sum of radii for maximization
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Optimizer constraints
    cons = []
    for i in range(n):
        # Bound constraints for the four directions, with closure-safe lambda
        # (Avoids lambda capturing i with the same value via capture using default arguments)
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints between circles, using closure safely
    # Use vector broadcasting to prevent O(n^2) constraint creation and ensure efficient evaluation
    for i in range(n):
        for j in range(i + 1, n):
            # Avoid lambda capture issues by using default arguments
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization pass with high precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-8})

    # If initial optimization fails, use an adaptive initialization vector
    if not res.success:
        # Apply adaptive initialization based on geometric hashing: use spatial clustering with perturbation
        # Introduce hierarchical geometric hashing with spatially dependent randomness
        hash_map = np.random.rand(n, 2)
        # Use a gradient-based adjustment to avoid bad initial clustering
        # Spatial hashing with adaptive scaling proportional to radius and position
        adaptive_hash = np.zeros((n, 2))
        for i in range(n):
            adaptive_hash[i, 0] = hash_map[i, 0] * (1.0 - v0[2::3][i] / r0) * np.cos(v0[0::3][i] * np.pi * 0.5)
            adaptive_hash[i, 1] = hash_map[i, 1] * (1.0 - v0[2::3][i] / r0) * np.sin(v0[1::3][i] * np.pi * 0.3)
        v0_reinit = v0.copy()
        v0_reinit[0::3] += adaptive_hash[:, 0]
        v0_reinit[1::3] += adaptive_hash[:, 1]
        # Second optimization pass with reinitialized points
        res = minimize(neg_sum_radii, v0_reinit, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 2000, "ftol": 1e-10, "eps": 2e-8})

    # Perform a targeted re-configuration phase: reordering circles to enhance radial growth
    # This phase focuses on least-restrained areas without violating constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Vectorized distance calculation using broadcasting for improved performance
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # Identify least-restrained circles using min distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Target least-restrained for expansion
        # Use adaptive expansion based on local density, not just global, to preserve constraints
        # Expand this circle first using a radial-based expansion that adjusts based on spatial configuration
        # Create perturbation vector and apply it in reverse direction to avoid constraints
        # Calculate allowed expansion for the least constrained circle
        # Ensure expansion is under the spatial constraints of nearby circles
        # Calculate allowable expansion for least constrained circle
        # To estimate maximum allowable radius for circle least_constrained_idx
        allowable_expansion = 0.0
        for j in range(n):
            if j == least_constrained_idx:
                continue
            # Compute the current distance and allowable expansion
            current_distance = np.sqrt((centers[least_constrained_idx, 0] - centers[j, 0])**2 + (centers[least_constrained_idx, 1] - centers[j, 1])**2)
            if current_distance == 0:
                # Skip if the circle would overlap itself; it's already constrained
                allowable_expansion = 0.0
                break
            # Maximum allowable radii for both circles given separation current_distance
            max_radius1 = (current_distance - radii[j]) / 2
            # Ensure expansion doesn't violate the maximum allowable radius for the target circle
            allowable_expansion = min(allowable_expansion, max_radius1 - radii[least_constrained_idx])
            # Also, ensure this circle doesn't exceed the boundary constraints
            max_boundary_radius = 0.5
            if centers[least_constrained_idx, 0] < 0.5:
                max_boundary_radius = min(max_boundary_radius, centers[least_constrained_idx, 0])
            else:
                max_boundary_radius = min(max_boundary_radius, 1.0 - centers[least_constrained_idx, 0])
            if centers[least_constrained_idx, 1] < 0.5:
                max_boundary_radius = min(max_boundary_radius, centers[least_constrained_idx, 1])
            else:
                max_boundary_radius = min(max_boundary_radius, 1.0 - centers[least_constrained_idx, 1])
            max_boundary_radius = np.minimum(max_boundary_radius, 0.5)  # Ensure not exceeding radius limits
            allowable_expansion = min(allowable_expansion, max_boundary_radius - radii[least_constrained_idx])
        # Apply expansion to the least constrained circle with safety margin
        expansion_amount = np.clip(allowable_expansion * 1.1, 1e-6, 0.005)  # Add safety margin to prevent constraint violations
        # Create a new vector and slightly alter positions to reconfigure the grid
        # For reconfiguration, apply a spatial reconfiguration that shifts the least constrained circle slightly for optimal growth
        # To ensure constraints, use a constrained optimization approach
        # Reconstruct v, alter the least constrained circle's radius, and perform a re-optimization
        # Create a new optimization vector where radius of least_constrained_idx is expanded
        v_new = v.copy()
        v_new[3 * least_constrained_idx + 2] += expansion_amount
        # Re-run optimization with this new vector and ensure the system remains valid
        # Use a lower maxiter to avoid unnecessary computations
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-9})
        if not res.success:
            # If reconfiguration fails, use a fallback that increases all radii by a small fraction
            # This avoids constraint breakage by using a more uniform expansion
            v_new = v.copy()
            delta = (1.0 - v[2::3] / r0) * 0.002  # Small uniform expansion
            v_new[2::3] += delta
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})

    # Final cleanup and return
    v = res.x if res.success else v0  # Use fallback if optimization fails
    # Clean up: clip out-of-bound radii
    radii = np.clip(v[2::3], 1e-6, 0.5)  # Ensure radii do not exceed the maximum possible in unit square
    centers = np.column_stack([v[0::3], v[1::3]])
    return centers, radii, float(radii.sum())