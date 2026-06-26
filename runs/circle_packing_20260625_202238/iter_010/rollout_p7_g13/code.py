import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

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

    # Initial optimization with standard setup
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Identify most restrictive constraint
    max_violation = -np.inf
    most_restricted_circle = -1
    for i in range(n):
        for j in range(i + 1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            violation = max(0, (v[3*i+2] + v[3*j+2] - dist))
            if violation > max_violation:
                max_violation = violation
                most_restricted_circle = i if i != j else -1
    
    if most_restricted_circle != -1:
        # Create a new initial guess with boundary radius modified by ±20%
        modified_v = v.copy()
        # Apply ±20% deviation to the radius of the most constrained circle
        modified_v[3*most_restricted_circle+2] *= 1.2
        modified_v[3*most_restricted_circle+2] = np.clip(modified_v[3*most_restricted_circle+2], 1e-4, 0.5)
        modified_v[3*most_restricted_circle+2] *= 0.8
        modified_v[3*most_restricted_circle+2] = np.clip(modified_v[3*most_restricted_circle+2], 1e-4, 0.5)
        
        # Re-optimize with modified initial guess
        res = minimize(neg_sum_radii, modified_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
        v = res.x if res.success else modified_v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)

    # Add penalty for overlapping circles to improve convergence
    def penalty(v):
        sum_penalty = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                    sum_penalty += max(0, (v[3*i+2] + v[3*j+2] - dist) ** 2)
        return sum_penalty

    # Re-seed using a Voronoi-based initial configuration
    # Generate a Voronoi tessellation and use its sites as initial centers
    from scipy.spatial import Voronoi
    # Create an initial set of points based on Voronoi tessellation in a square
    initial_voronoi_points = np.random.rand(n, 2)
    vor = Voronoi(initial_voronoi_points)
    vor_centers = np.array([vor.vertices[i] for i in vor.point_region])
    vor_centers = (vor_centers + 1) / 2  # Normalize to [0,1] square
    # Create a new initial guess based on Voronoi points
    vor_v = np.empty(3 * n)
    vor_v[0::3] = vor_centers[:, 0]
    vor_v[1::3] = vor_centers[:, 1]
    vor_v[2::3] = r0

    # Re-optimize with Voronoi-based initial guess
    res = minimize(neg_sum_radii, vor_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else vor_v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Jiggle heuristic for smallest circles
    if np.sum(radii) > 0:
        # Sort circles by radius (smallest first)
        sorted_indices = np.argsort(radii)
        # Select the smallest 10 circles
        small_circle_indices = sorted_indices[:10]
        # Perturb their positions slightly and re-optimize
        perturbation = 0.01
        for idx in small_circle_indices:
            i = idx
            v[3*i] += np.random.uniform(-perturbation, perturbation)
            v[3*i+1] += np.random.uniform(-perturbation, perturbation)
            v[3*i+2] = np.clip(v[3*i+2], 1e-6, 0.5)
        # Re-optimize with penalty
        res = minimize(lambda v: -np.sum(v[2::3]) + 100 * penalty(v), v, method="SLSQP", 
                       bounds=bounds, constraints=cons, 
                       options={"maxiter": 300, "ftol": 1e-9})
        v = res.x if res.success else v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())