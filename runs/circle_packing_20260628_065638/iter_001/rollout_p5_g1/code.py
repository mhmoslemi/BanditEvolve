import numpy as np

def run_packing():
    n = 26
    # Use hexagonal grid seeding for better initial packing
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    # Generate hexagonal grid points
    points = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5 * (i % 2)) / cols
            y = i / rows
            points.append((x, y))
            if len(points) == n:
                break
        if len(points) == n:
            break
    # Initial radii: based on average spacing
    r0 = 0.2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array([p[0] for p in points])
    v0[1::3] = np.array([p[1] for p in points])
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Define constraints for boundaries and circle radii
    def boundary_constraints(v, i):
        x, y, r = v[3*i], v[3*i+1], v[3*i+2]
        return np.array([
            x - r,        # x - r >= 0
            1.0 - x - r,  # x + r <= 1
            y - r,        # y - r >= 0
            1.0 - y - r   # y + r <= 1
        ])

    # Define constraints for circle overlaps
    def overlap_constraints(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        r_i = v[3*i+2]
        r_j = v[3*j+2]
        return dx*dx + dy*dy - (r_i + r_j)**2

    # First stage: Coarse global search using SLSQP
    cons_global = []
    for i in range(n):
        for expr in boundary_constraints(v0, i):
            cons_global.append({"type": "ineq", "fun": lambda v, i=i: expr})
    for i in range(n):
        for j in range(i + 1, n):
            cons_global.append({"type": "ineq", "fun": lambda v, i=i, j=j: overlap_constraints(v, i, j)})
    
    res_global = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                         constraints=cons_global, options={"maxiter": 200, "ftol": 1e-8})

    v = res_global.x if res_global.success else v0

    # Second stage: Local optimization using L-BFGS-B for better convergence
    cons_local = []
    for i in range(n):
        for expr in boundary_constraints(v, i):
            cons_local.append({"type": "ineq", "fun": lambda v, i=i: expr})
    for i in range(n):
        for j in range(i + 1, n):
            cons_local.append({"type": "ineq", "fun": lambda v, i=i, j=j: overlap_constraints(v, i, j)})
    
    res_local = minimize(neg_sum_radii, v, method="L-BFGS-B", bounds=bounds,
                         constraints=cons_local, options={"maxiter": 300, "ftol": 1e-9})

    v = res_local.x if res_local.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())