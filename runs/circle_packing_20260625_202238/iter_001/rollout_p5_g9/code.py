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

    # Generate initial constraint list with all pairwise circles
    for i in range(n):
        for j in range(i + 1, n):
            # Use delayed evaluation to avoid closure capture issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Evaluate constraint tightness and reorder for mutation
    def constraint_tightness(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        tightness = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                tightness.append((dist - (radii[i] + radii[j]), i, j))
        # Sort by tightness (most tight first)
        tightness.sort()
        return tightness

    tightness_list = constraint_tightness(v)
    # Reorder indices based on constraint tightness
    reordered = [0] * n
    for idx, (t, i, j) in enumerate(tightness_list):
        reordered[i] = idx
        reordered[j] = idx
    # Create new ordering for all circles
    new_order = np.argsort(reordered)

    # Reorder the decision vector and constraints
    def reorder_vector(v, new_order):
        new_v = np.empty_like(v)
        for i, j in enumerate(new_order):
            new_v[3*i] = v[3*j]
            new_v[3*i+1] = v[3*j+1]
            new_v[3*i+2] = v[3*j+2]
        return new_v

    def reorder_constraints(cons, new_order):
        new_cons = []
        for c in cons:
            if "fun" in c and callable(c["fun"]):
                def new_fun(v, i=new_order.index(c["fun"].__code__.co_freevars[0]), j=new_order.index(c["fun"].__code__.co_freevars[1])):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                new_cons.append({"type": "ineq", "fun": new_fun})
            else:
                new_cons.append(c)
        return new_cons

    # Perform mutation by reordering
    v_mutated = reorder_vector(v, new_order)
    cons_mutated = reorder_constraints(cons, new_order)

    # Second optimization with mutated constraints
    res_mutated = minimize(neg_sum_radii, v_mutated, method="SLSQP", bounds=bounds,
                           constraints=cons_mutated, options={"maxiter": 500, "ftol": 1e-9})
    v_mutated = res_mutated.x if res_mutated.success else v_mutated

    # Local polishing step to resolve overlapping and out-of-bounds issues
    def polish_solution(v):
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-12:
                    # Adjust the smaller radius to avoid overlap
                    if radii[i] < radii[j]:
                        radii[i] = dist - radii[j] + 1e-12
                    else:
                        radii[j] = dist - radii[i] + 1e-12
        # Adjust centers to keep within bounds
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0:
                centers[i, 0] = r
            elif x + r > 1:
                centers[i, 0] = 1 - r
            if y - r < 0:
                centers[i, 1] = r
            elif y + r > 1:
                centers[i, 1] = 1 - r
        # Update v
        v[0::3] = centers[:, 0]
        v[1::3] = centers[:, 1]
        v[2::3] = radii
        return v

    v_mutated = polish_solution(v_mutated)

    centers = np.column_stack([v_mutated[0::3], v_mutated[1::3]])
    radii = np.clip(v_mutated[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())