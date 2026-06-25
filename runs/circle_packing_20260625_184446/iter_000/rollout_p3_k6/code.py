import numpy as np

def run_packing():
    n = 26
    # Initial guess: grid-based arrangement
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = 0.5 / cols - 1e-3

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
            def constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint})

    # Use differential evolution for better exploration
    from scipy.optimize import differential_evolution
    def objective(v):
        return -np.sum(v[2::3])

    def constraints(v):
        constraints = []
        for i in range(n):
            dx = v[3*i] - v[3*i+2]
            constraints.append(dx)
            constraints.append(1.0 - v[3*i] - v[3*i+2])
            dy = v[3*i+1] - v[3*i+2]
            constraints.append(dy)
            constraints.append(1.0 - v[3*i+1] - v[3*i+2])
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                constraints.append(dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2)
        return constraints

    # Format constraints for differential evolution
    def constraint_func(v):
        cons = []
        for i in range(n):
            dx = v[3*i] - v[3*i+2]
            cons.append(dx)
            cons.append(1.0 - v[3*i] - v[3*i+2])
            dy = v[3*i+1] - v[3*i+2]
            cons.append(dy)
            cons.append(1.0 - v[3*i+1] - v[3*i+2])
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                cons.append(dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2)
        return np.array(cons)

    result = differential_evolution(
        objective,
        bounds,
        constraints={"type": "ineq", "fun": constraint_func},
        popsize=15,
        maxiter=200,
        tol=1e-8,
        mutation=(0.5, 1),
        recombination=0.9
    )

    v = result.x
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())