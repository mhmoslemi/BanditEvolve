import numpy as np

def run_packing():
    n = 26
    # Optimize spatial clustering to balance between grid spacing and dynamic adjustment
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with an adaptive grid: alternating row spacing and radial adjustment
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.3) / cols  # Slightly off-center to avoid symmetry
        base_y = (row + 0.3) / rows  # Slightly off-center vertically
        
        # Add jitter to avoid convergence on symmetric solutions
        jitter = np.random.uniform(-0.04, 0.04)
        x_offset = (np.sin(i * np.pi / 7) * 0.012) + np.random.uniform(-0.002, 0.002)
        y_offset = (np.cos(i * np.pi / 7) * 0.012) + np.random.uniform(-0.002, 0.002)
        
        x = base_x + x_offset
        y = base_y + y_offset
        
        # Shift alternate rows to introduce vertical disparity and reduce symmetry
        if row % 2 == 1:  # Odd rows (counting from 0)
            y += np.sqrt(3) / (2 * cols)  # Offset vertically more to stagger rows
        
        # Edge cases: ensure boundaries are respected by initial jittering
        if x < 0.005 or x > 1 - 0.005:
            x = max(0.005, min(1 - 0.005, x))
        if y < 0.005 or y > 1 - 0.005:
            y = max(0.005, min(1 - 0.005, y))
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius setup based on grid spacing and spatial distribution
    # Use dynamic radius allocation: grid-based but also influenced by inter-cell distance
    # We calculate an average initial radius that can be scaled based on spatial distribution
    avg_initial_radius_base = 0.35 / cols - 1e-3
    r0 = []
    
    # For better control, we calculate a baseline that depends on average cell spacing
    # First, estimate the initial average cell spacing (diagonal distance between nearest neighbors)
    # Using a grid spacing based on cols and rows
    avg_cell_diagonal = np.sqrt((cols)**-2 + (rows)**-2)  # In units of [0,1]
    # We'll use a safe but efficient value based on this, then optimize
    r0 = [avg_initial_radius_base * (1.0 + (0.3 * (1.0 / np.sqrt(2)) / avg_cell_diagonal)) ** 2]
    r0 = [r0[0]] * n  # For now uniform, but later may be refined
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.array(r0)

    # Bounds: ensure centers are in [0, 1], radii in [1e-4, 0.5]
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n, matches v

    # Objective: maximize sum of radii = minimize -sum(radii)
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Boundary constraints
    cons = []
    # Each circle's center must be at least radius away from square edges
    for i in range(n):
        # Left edge constraint: x - r >= 0 → x - r ≥ 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right edge constraint: x + r <= 1 → 1 - x - r ≥ 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom edge constraint: y - r >= 0 → y - r ≥ 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top edge constraint: y + r <= 1 → 1 - y - r ≥ 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Overlap constraint between any two distinct circles
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First phase: initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "eps": 1e-8})

    # If we found a solution, we refine it with more targeted steps
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute a more accurate radius distribution based on current spatial context
        # First, calculate inter-circle distances and current effective constraints
        # Use broadcasting for vectorized distance calculations
        dx_full = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy_full = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx_full ** 2 + dy_full ** 2)
        
        # Find each circle's minimum distance to others for constraint strength
        min_dists = np.min(dists, axis=1)
        
        # For each circle, estimate max additional radius it can get without overlapping
        # Based on current minimal distance to others, and the current minimal distance to the boundary
        possible_radii = []
        for idx in range(n):
            current_r = radii[idx]
            # Calculate minimal distance to square sides
            min_bound_dist = np.min([v[3*idx] - current_r, 1 - v[3*idx] - current_r,
                                   v[3*idx + 1] - current_r, 1 - v[3*idx + 1] - current_r])
            
            # Distance to neighbors
            min_neigh_dist = np.min(dists[idx, :]) if idx < n else 0
            
            # Max additional radius it can expand without overlapping, assuming it remains fixed
            max_add_radius = min(np.min([min_bound_dist, min_neigh_dist / 2 - current_r]))
            # Apply slight constraint on expansion for better numerical behavior
            # Prevent excessive expansion that could destabilize the system
            possible_radii.append(max_add_radius)  # We'll use this in later steps, not apply immediately

        # Now, we create a perturbation grid with adaptive scaling based on current distances
        # We generate a random perturbation field, but scaled by the minimal distance to neighbors
        # To avoid symmetry and improve local exploration, we use different per-circle scaling
        random_perturbation = np.random.rand(n, 2) * (0.05)  # Max 5% perturbation
        # Scale perturbations by distance to neighbors to improve convergence
        scaling_factor = (min_dists / (min_dists.mean() + 1e-4)) ** 0.5  # sqrt for less aggressive scaling
        # Apply perturbations to positions in a way that keeps them within [0, 1]
        for i in range(n):
            pert_x = random_perturbation[i, 0] * scaling_factor[i]
            pert_y = random_perturbation[i, 1] * scaling_factor[i]
            v[3*i] += pert_x
            v[3*i + 1] += pert_y
            
            # Clamp positions to unit square (with buffer)
            v[3*i] = np.clip(v[3*i], 0.005, 0.995)
            v[3*i + 1] = np.clip(v[3*i + 1], 0.005, 0.995)
        
        # Second phase: perturbed optimization to avoid converging on same local minima
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-11, "eps": 1e-8})
    
    # If still successful, do post-optimization targeted expansions
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute distances again for accurate constraint usage
        dx_full = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy_full = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx_full ** 2 + dy_full ** 2)
        
        # Identify the circle with the largest margin to boundary and others
        # This is our "least constrained" circle, which can expand the most without destabilizing
        # For each circle, find the maximum possible expansion it can undergo (based on current layout)
        # Calculate for each circle:
        # 1. Minimum distance to boundaries
        # 2. Minimum distance to other circles
        # 3. The maximum possible radius increase it can take without conflict
        max_radius_increase = []
        for i in range(n):
            current_r = radii[i]
            # Distance to boundaries
            x, y = centers[i]
            bound_dist = min(x - current_r, 1 - x - current_r, y - current_r, 1 - y - current_r)
            # Distance to others
            min_ngh_dist = dists[i].min() if i < n else 0
            # Max that can be added without causing conflict
            max_addable = min(bound_dist, (min_ngh_dist - current_r) / 2 if min_ngh_dist > current_r else np.inf)
            max_radius_increase.append(max_addable)
        
        # Find the circle with the largest room for expansion
        # This is our target for expansion
        best_idx = np.argmax(max_radius_increase)
        best_possible_increase = max_radius_increase[best_idx]
        
        # Also, evaluate the circle with the smallest radius - it may be able to grow if it's on a "spine" of the configuration
        smallest_radius_idx = np.argmin(radii)
        smallest_radius = radii[smallest_radius_idx]
        min_ngh_of_smaller = dists[smallest_radius_idx].min()

        # Calculate how much we can increase the minimum-radius circle
        # We'll expand it slightly to potentially unlock new geometric configurations
        # Use a factor based on how much it can grow without violating constraints
        # We'll also use a small expansion for the circle with the largest room
        # We'll do this carefully to avoid over-expanding
        expansion_factor = (best_possible_increase + (min_ngh_of_smaller - smallest_radius) / 2) * 0.25
        if expansion_factor < 0:
            expansion_factor = 0

        # To improve stability, we use a small but targeted expansion
        # Rather than expanding all, we focus on best_idx and smallest_radius_idx
        # We'll first try to grow the best_idx circle
        new_radii = radii.copy()
        # Slightly increase the best circle (with most room) and the smallest
        new_radii[best_idx] += expansion_factor * 1.2  # Slight over-expansion to potentially unlock new configurations
        new_radii[smallest_radius_idx] += expansion_factor * 0.8  # Less aggressive for the smaller
        # For others, slightly increase radius as a catalyst to trigger layout shifts
        for i in range(n):
            if i not in [best_idx, smallest_radius_idx]:
                new_radii[i] += expansion_factor * 0.3  # Slight expansion for others

        # Rebuild v with the updated radii
        v_new = v.copy()
        v_new[2::3] = new_radii

        # Validate the new configuration before final optimization
        # Use a pre-validation pass to prevent invalid configurations
        # This is an early check to ensure our new setup is valid
        valid, val_msg = validate_packing(np.column_stack([v_new[0::3], v_new[1::3]]), v_new[2::3])
        if valid:
            # Final optimization with this new configuration
            # Use a slightly more aggressive configuration
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
        else:
            # If invalid, revert to a safer state with only the most constrained circle expanded
            # We apply expansion only to the best_idx and smallest_radius_idx in a limited way, ensuring it's valid
            # We will do a cautious expansion of only these two to avoid invalid state
            # Rebuild radii with minimal increase and re-validate
            # Only increase best_idx and smallest_radius_idx by a fraction of allowed limit
            safe_increase = 0.7 * best_possible_increase  # Safe increase for best circle
            safe_increase_smallest = 0.2 * (min_ngh_of_smallest - smallest_radius) / 2
            new_radii = radii.copy()
            new_radii[best_idx] += safe_increase
            new_radii[smallest_radius_idx] += safe_increase_smallest
            v_new = v.copy()
            v_new[2::3] = new_radii
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})

    # After all optimizations, ensure we do not proceed with invalid states
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    # Final clipping to ensure no negative radii (due to numerical imprecision)
    radii = np.clip(v[2::3], 1e-6, None)

    # Return the centers, radii (as per specification), and sum of radii
    return centers, radii, float(radii.sum())