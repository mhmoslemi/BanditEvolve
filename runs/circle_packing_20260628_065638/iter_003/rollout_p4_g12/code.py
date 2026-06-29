import numpy as np

def run_packing():
    n = 26
    cols = 5  # Hexagonal grid with 5 columns
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a hexagonal grid pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Coarse global optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Local refinement with L-BFGS-B
    if res.success:
        # Use the current radius values for refinement
        def refine_neg_sum_radii(v):
            return -np.sum(v[2::3])
        
        # Create a new optimization problem with the same constraints
        res_refine = minimize(refine_neg_sum_radii, v, method="L-BFGS-B", bounds=bounds,
                             constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
        v = res_refine.x if res_refine.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Final local polishing step
    if np.sum(radii) > 0:
        max_radius_index = np.argmax(radii)
        max_radius = radii[max_radius_index]
        # Check if we can increase the radius of the largest circle
        # by a small amount without causing overlap
        for i in range(n):
            if i == max_radius_index:
                continue
            dx = centers[max_radius_index, 0] - centers[i, 0]
            dy = centers[max_radius_index, 1] - centers[i, 1]
            min_dist = np.sqrt(dx*dx + dy*dy)
            if min_dist < radii[max_radius_index] + radii[i] - 1e-8:
                break
        else:
            # No overlap, safely increase the radius
            radii[max_radius_index] += 0.001
            # Ensure the new radius does not exceed the square's boundaries
            if centers[max_radius_index, 0] - radii[max_radius_index] < 0:
                radii[max_radius_index] = centers[max_radius_index, 0]
            if centers[max_radius_index, 0] + radii[max_radius_index] > 1:
                radii[max_radius_index] = 1 - centers[max_radius_index, 0]
            if centers[max_radius_index, 1] - radii[max_radius_index] < 0:
                radii[max_radius_index] = centers[max_radius_index, 1]
            if centers[max_radius_index, 1] + radii[max_radius_index] > 1:
                radii[max_radius_index] = 1 - centers[max_radius_index, 1]

    return centers, radii, float(radii.sum())