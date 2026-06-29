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

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    
    # If optimization fails, refine with local polishing
    if not res.success:
        v = v0
    else:
        v = res.x

    # Local polishing step to refine the solution
    def local_refine(v):
        v_local = v.copy()
        for _ in range(100):
            # Update positions and radii using gradient descent
            grad = np.zeros_like(v_local)
            for i in range(n):
                # Gradient of objective function
                grad[3*i+2] = -1.0  # negative of gradient of -sum(radii)
                
                # Gradient of constraints for boundaries
                for c in cons:
                    if c["type"] == "ineq":
                        # For boundary constraints, derivative with respect to x, y, r
                        if c["fun"](v_local) < 0:
                            # Constraint is active, so gradient is non-zero
                            if c["fun"].__name__ == "lambda v, i=i: v[3*i] - v[3*i+2]":
                                grad[3*i] += 1.0
                                grad[3*i+2] -= 1.0
                            elif c["fun"].__name__ == "lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]":
                                grad[3*i] += 1.0
                                grad[3*i+2] -= 1.0
                            elif c["fun"].__name__ == "lambda v, i=i: v[3*i+1] - v[3*i+2]":
                                grad[3*i+1] += 1.0
                                grad[3*i+2] -= 1.0
                            elif c["fun"].__name__ == "lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]":
                                grad[3*i+1] += 1.0
                                grad[3*i+2] -= 1.0
                            elif "constraint_func" in c["fun"].__name__:
                                # For distance constraints
                                i_c, j_c = c["fun"].__name__.split("_")[1:], c["fun"].__name__.split("_")[3:]
                                i_c, j_c = int(i_c[0]), int(j_c[0])
                                dx = v_local[3*i_c] - v_local[3*j_c]
                                dy = v_local[3*i_c+1] - v_local[3*j_c+1]
                                dist_sq = dx*dx + dy*dy
                                min_dist_sq = (v_local[3*i_c+2] + v_local[3*j_c+2])**2
                                if dist_sq < min_dist_sq:
                                    grad[3*i_c] += 2 * dx / (2 * np.sqrt(dist_sq))
                                    grad[3*i_c+1] += 2 * dy / (2 * np.sqrt(dist_sq))
                                    grad[3*i_c+2] -= 2 * (v_local[3*i_c+2] + v_local[3*j_c+2]) / (2 * np.sqrt(dist_sq))
                                    grad[3*j_c] += -2 * dx / (2 * np.sqrt(dist_sq))
                                    grad[3*j_c+1] += -2 * dy / (2 * np.sqrt(dist_sq))
                                    grad[3*j_c+2] -= 2 * (v_local[3*i_c+2] + v_local[3*j_c+2]) / (2 * np.sqrt(dist_sq))
            # Update v_local
            v_local -= 0.01 * grad
            # Enforce bounds
            v_local[::3] = np.clip(v_local[::3], 0.0, 1.0)
            v_local[1::3] = np.clip(v_local[1::3], 0.0, 1.0)
            v_local[2::3] = np.clip(v_local[2::3], 1e-4, 0.5)
        return v_local

    v = local_refine(v)
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())