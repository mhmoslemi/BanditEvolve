import numpy as np

def run_packing():
    n = 26
    cols = 6  # Increase rows to better handle more circles with more efficient packing
    rows = (n + cols - 1) // cols
    
    # Initialize position via improved geometric seeding with grid + dynamic jitter
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Introduce asymmetric jittering to break symmetry and optimize space
        x_offset = np.random.uniform(-0.03, 0.03) * (1 + 0.5 * (row % 3 == 0))
        y_offset = np.random.uniform(-0.03, 0.03) * (1 + 0.5 * (col % 5 == 0))
        # Create staggered grid by shifting even rows slightly
        if row % 2 == 0:
            x_center += 0.5 / cols * (1 - 0.5 * np.random.rand())
        x = x_center + x_offset
        y = y_center + y_offset
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3  # Higher initial radius than previous to kickstart packing
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure 3n bounds

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Minimize negative of sum to maximize radius sum

    # Vectorized constraint creation for boundaries using lambda with captured i
    cons = []
    for i in range(n):
        # Left wall constraint: x >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right wall constraint: 1 - x <= r (x + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom wall constraint: y >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top wall constraint: 1 - y <= r (y + r <= 1)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized distance-based constraint for non-overlapping (i,j) pairs
    for i in range(n):
        for j in range(i + 1, n):
            # Apply jitter-robust closure with explicit i and j capture
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j]) ** 2 + 
                    (v[3*i+1] - v[3*j+1]) ** 2 - 
                    (v[3*i+2] + v[3*j+2]) ** 2
                )
            })

    # Initial phase: basic constrained optimization with high tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})

    # Secondary phase: perturbation + refinement (shake heuristic)
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]

        # Shaking phase: perturb small circles to escape shallow local optima
        # Identify circles with small radii for aggressive shaking
        small_radius_idx = np.where(radii < np.mean(radii) * 0.7)[0]
        if len(small_radius_idx) > 0:
            # Apply controlled spatial jitter to break constraints
            for idx in small_radius_idx:
                # Apply spatial jitter in the direction opposite to nearest neighbor
                nearest_dist = np.inf
                nearest_idx = -1
                for j in range(n):
                    if j == idx:
                        continue
                    dx = centers[idx, 0] - centers[j, 0]
                    dy = centers[idx, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_idx = j
                if nearest_idx == -1:
                    continue
                # Compute jitter vector to avoid overlap with nearest neighbor
                dx_jitter = (radii[idx] + radii[nearest_idx]) / (np.sqrt(dx**2 + dy**2) + 1e-10)
                dx_jitter *= (dx / nearest_dist)
                dy_jitter = (radii[idx] + radii[nearest_idx]) / (np.sqrt(dx**2 + dy**2) + 1e-10)
                dy_jitter *= (dy / nearest_dist)
                # Apply jitter
                v[3*idx] += dx_jitter * 0.1
                v[3*idx+1] += dy_jitter * 0.1

        # Refine the perturbed solution with tighter tolerances
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1200, "ftol": 1e-11})

    # Final phase: adaptive radius expansion with gradient-directed search
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle by minimizing total overlap potential
        total_overlap_potential = np.sum((dists - (radii[np.newaxis, :] + radii[:, np.newaxis])) ** 2, axis=1)
        least_constrained_idx = np.argmin(total_overlap_potential)
        
        # Calculate growth based on current total sum and expansion potential
        current_total = np.sum(radii)
        # Expand by 0.01 (double 0.005), with dynamic allocation
        expansion = 0.01
        # Create expansion vector with targeted expansion on the least constrained circle
        # Use softmax to focus more expansion on underutilized circles
        expansion_factors = np.exp(-total_overlap_potential * 0.01)
        expansion_factors /= expansion_factors.sum()
        expansion_factors[least_constrained_idx] *= 2.0  # Double for aggressive growth
        
        new_radii = radii.copy()
        new_radii += expansion_factors * (expansion / (np.sum(expansion_factors) + 1e-10))
        # Clip to safety for overflow
        new_radii = np.clip(new_radii, 1e-7, 0.49)
        
        # Apply expansion and re-optimize
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1200, "ftol": 1e-11})

    # Final cleanup
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())