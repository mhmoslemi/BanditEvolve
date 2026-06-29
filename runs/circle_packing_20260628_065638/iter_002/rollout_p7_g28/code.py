import numpy as np

def run_packing():
    n = 26
    cols = 5  # Manual adjustment for a hexagonal grid
    rows = (n + cols - 1) // cols  # Ensure enough rows for 26 circles
    
    # Initialize positions using a hexagonal grid pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Offset even rows for hexagonal packing
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

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Hybrid mutation: increase the radius of the largest circle if possible
    if res.success:
        max_idx = np.argmax(radii)
        max_radius = radii[max_idx]
        # Check for overlap after increasing the radius
        for j in range(n):
            if j == max_idx:
                continue
            dx = centers[max_idx, 0] - centers[j, 0]
            dy = centers[max_idx, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < max_radius + radii[j] - 1e-12:
                continue  # Cannot increase without overlap
        # Try to increase the radius by a small amount
        radii[max_idx] += 0.001
        # Adjust other circles to avoid overlap
        for j in range(n):
            if j == max_idx:
                continue
            dx = centers[max_idx, 0] - centers[j, 0]
            dy = centers[max_idx, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[max_idx] + radii[j] - 1e-12:
                # Move the smaller circle slightly to avoid overlap
                overlap = radii[max_idx] + radii[j] - dist
                move = overlap * 0.5
                angle = np.arctan2(dy, dx)
                centers[j, 0] += np.cos(angle) * move
                centers[j, 1] += np.sin(angle) * move

    return centers, radii, float(radii.sum())