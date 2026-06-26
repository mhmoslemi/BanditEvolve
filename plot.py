"""
Visualize a circle packing produced by a run_packing() function.

Usage:
    1. Paste your run_packing() below where indicated
    2. python plot_packing.py
       (or python plot_packing.py output.png   to save instead of display)

Validates the packing the same way reward.py does and prints any issues.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize
import numpy.random as npr

import numpy as np
import math
import random
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

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

    # Vectorized overlap constraints
    def create_overlap_constraints():
        overlap_cons = []
        for i in range(n):
            for j in range(i + 1, n):
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                overlap_cons.append({"type": "ineq", "fun": constraint_func})
        return overlap_cons

    cons += create_overlap_constraints()

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Compute constraint tightness
    constraint_tightness = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                constraint_tightness[i] += (v[3*i+2] + v[3*j+2] - dist)
                constraint_tightness[j] += (v[3*i+2] + v[3*j+2] - dist)
    
    # Sort indices by constraint tightness (most constrained first)
    sorted_indices = np.argsort(constraint_tightness)
    
    # Permute the decision vector based on sorted indices
    permuted_v = np.zeros_like(v)
    for i, idx in enumerate(sorted_indices):
        permuted_v[3*i] = v[3*idx]
        permuted_v[3*i+1] = v[3*idx+1]
        permuted_v[3*i+2] = v[3*idx+2]
    
    # Re-optimize with permuted initial guess
    res = minimize(neg_sum_radii, permuted_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else permuted_v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Apply dual-phase geometric distortion
    # Phase 1: Logarithmic scaling of coordinates
    log_v = np.log(v + 1e-10)
    log_v[0::3] = (log_v[0::3] - np.min(log_v[0::3])) / (np.max(log_v[0::3]) - np.min(log_v[0::3]))
    log_v[1::3] = (log_v[1::3] - np.min(log_v[1::3])) / (np.max(log_v[1::3]) - np.min(log_v[1::3]))
    log_v[2::3] = (log_v[2::3] - np.min(log_v[2::3])) / (np.max(log_v[2::3]) - np.min(log_v[2::3]))
    
    # Phase 2: Re-seed with distorted coordinates
    distorted_v = np.copy(log_v)
    distorted_v[0::3] *= 1.2
    distorted_v[1::3] *= 1.2
    distorted_v[2::3] *= 0.8

    # Re-optimize with distorted initial guess
    res = minimize(neg_sum_radii, distorted_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else distorted_v
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

    # Re-optimize with penalty
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v
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

    # Final cleanup pass: attempt to increase radii slightly without moving centers
    # This step is only performed if the current solution is valid and does not cause overlap
    if validate_packing(centers, radii)[0]:
        for i in range(n):
            # Try to increase radius by a small epsilon
            new_radius = radii[i] + 1e-6
            # Check if increasing this radius would cause overlap with any other circle
            overlap = False
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < new_radius + radii[j] - 1e-12:
                    overlap = True
                    break
            if not overlap:
                # Increase the radius and update the decision vector
                v[3*i+2] = new_radius
                # Update the radii and centers
                radii = np.clip(v[2::3], 1e-6, None)
                centers = np.column_stack([v[0::3], v[1::3]])

    # Apply geometric phase shift mutation
    # Isolate and invert the smallest circles
    sorted_indices = np.argsort(radii)
    small_circle_indices = sorted_indices[:3]
    modified_v = np.copy(v)
    for idx in small_circle_indices:
        i = idx
        modified_v[3*i] = 1.0 - v[3*i]
        modified_v[3*i+1] = 1.0 - v[3*i+1]
    res = minimize(neg_sum_radii, modified_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    modified_v = res.x if res.success else modified_v
    modified_centers = np.column_stack([modified_v[0::3], modified_v[1::3]])
    modified_radii = np.clip(modified_v[2::3], 1e-6, None)

    if validate_packing(modified_centers, modified_radii)[0] and np.sum(modified_radii) > np.sum(radii):
        v = modified_v
        centers = modified_centers
        radii = modified_radii

    # Final cleanup pass: attempt to increase radii slightly without moving centers
    # This step is only performed if the current solution is valid and does not cause overlap
    if validate_packing(centers, radii)[0]:
        for i in range(n):
            # Try to increase radius by a small epsilon
            new_radius = radii[i] + 1e-6
            # Check if increasing this radius would cause overlap with any other circle
            overlap = False
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < new_radius + radii[j] - 1e-12:
                    overlap = True
                    break
            if not overlap:
                # Increase the radius and update the decision vector
                v[3*i+2] = new_radius
                # Update the radii and centers
                radii = np.clip(v[2::3], 1e-6, None)
                centers = np.column_stack([v[0::3], v[1::3]])

    # Final refinement: fine-tune tolerances and re-optimize
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-12, "eps": 1e-12})
    v = res.x if res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Apply radical geometric inversion: isolate the three smallest circles
    sorted_indices = np.argsort(radii)
    small_circle_indices = sorted_indices[:3]
    # Swap positions of the smallest circles
    for i in range(len(small_circle_indices)):
        for j in range(i + 1, len(small_circle_indices)):
            idx_i = small_circle_indices[i]
            idx_j = small_circle_indices[j]
            # Swap x and y positions
            v[3*idx_i], v[3*idx_j] = v[3*idx_j], v[3*idx_i]
            v[3*idx_i+1], v[3*idx_j+1] = v[3*idx_j+1], v[3*idx_i+1]

    # Re-optimize with modified initial guess using a weighted objective
    def weighted_neg_sum_radii(v):
        weighted_sum = 0
        for i in range(n):
            # Weight small circles by 1.5 and others by 1.0
            if i in small_circle_indices:
                weighted_sum -= 1.5 * v[3*i+2]
            else:
                weighted_sum -= v[3*i+2]
        return weighted_sum

    res = minimize(weighted_neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Validate the modified solution and choose the better one
    if validate_packing(centers, radii)[0] and np.sum(radii) > np.sum(radii):
        v = v
        centers = centers
        radii = radii

    # Final cleanup pass: attempt to increase radii slightly without moving centers
    # This step is only performed if the current solution is valid and does not cause overlap
    if validate_packing(centers, radii)[0]:
        for i in range(n):
            # Try to increase radius by a small epsilon
            new_radius = radii[i] + 1e-6
            # Check if increasing this radius would cause overlap with any other circle
            overlap = False
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < new_radius + radii[j] - 1e-12:
                    overlap = True
                    break
            if not overlap:
                # Increase the radius and update the decision vector
                v[3*i+2] = new_radius
                # Update the radii and centers
                radii = np.clip(v[2::3], 1e-6, None)
                centers = np.column_stack([v[0::3], v[1::3]])

    return centers, radii, float(radii.sum())

# ====================================================================
# Validation (same as reward.py)
# ====================================================================
def validate_packing(centers, radii):
    """
    Paper-compatible signature: returns (bool, str).
    Matches reward.py so model code calling validate_packing(...) works.
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False, "NaN values present"
    for i in range(n):
        if radii[i] < 0:
            return False, f"Circle {i} has negative radius {radii[i]}"
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12
                or y - r < -1e-12 or y + r > 1 + 1e-12):
            return False, f"Circle {i} at ({x},{y}) r={r} outside unit square"
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.sqrt(np.sum((centers[i] - centers[j]) ** 2)))
            if dist < radii[i] + radii[j] - 1e-12:
                return False, f"Circles {i} and {j} overlap"
    return True, "ok"


def collect_issues(centers, radii, tol=1e-9):
    """Plotter-only helper. Returns a list of strings, one per problem found."""
    issues = []
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        issues.append("NaN values present")
    for i in range(n):
        if radii[i] < 0:
            issues.append(f"Circle {i} has negative radius {radii[i]:.6f}")
        x, y = centers[i]
        r = radii[i]
        if (x - r < -tol or x + r > 1 + tol
                or y - r < -tol or y + r > 1 + tol):
            issues.append(f"Circle {i} at ({x:.4f},{y:.4f}) r={r:.4f} outside unit square")
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.sqrt(np.sum((centers[i] - centers[j]) ** 2)))
            gap = dist - (radii[i] + radii[j])
            if gap < -tol:
                issues.append(f"Circles {i},{j} overlap (gap={gap:.6f})")
    return issues


# ====================================================================
# Plotting
# ====================================================================
def plot_packing(centers, radii, sum_radii, save_to=None):
    n = len(radii)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Unit square
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, linewidth=1.5, edgecolor="black"))

    # Color by radius so it's easy to see who's big and who's tiny
    cmap = plt.get_cmap("viridis")
    rmax = max(radii.max(), 1e-9)
    issues = collect_issues(centers, radii)
    invalid_ids = set()
    for msg in issues:
        # Best-effort: pull circle indices out of error messages so we can flag them
        for tok in msg.replace(",", " ").split():
            if tok.isdigit():
                invalid_ids.add(int(tok))

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        color = cmap(r / rmax)
        edge = "red" if i in invalid_ids else "black"
        lw = 1.5 if i in invalid_ids else 0.5
        ax.add_patch(patches.Circle((x, y), r,
                                     facecolor=color, edgecolor=edge,
                                     linewidth=lw, alpha=0.65))
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=8, color="white", weight="bold")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(True, alpha=0.3)

    title = f"n={n}  sum of radii = {sum_radii:.10f}, SOTA = 2.635983"
    if issues:
        title += f"  [INVALID: {len(issues)} issue(s)]"
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_to}")
    else:
        plt.show()


# How the sandbox scores a packing (CirclePacking.score in problems/circle_packing.py):
# valid + non-degenerate -> reward = float(np.sum(radii)); otherwise fail_score.
SEED = 0              # evaluation seed (num_eval_seeds=1 -> eval seed 0)
FAIL_SCORE = 0.0      # problem.fail_score
MIN_SUM_RADII = 1e-3  # CirclePacking.min_sum_radii (degenerate guard)


def sandbox_score(centers, radii):
    """Reproduce CirclePacking.score(): np.sum(radii) if the packing validates
    and is not degenerate, else the fail score (0.0)."""
    valid, msg = validate_packing(centers, radii)
    if not valid:
        return FAIL_SCORE, msg
    s = float(np.sum(radii))
    if s < MIN_SUM_RADII:
        return FAIL_SCORE, f"degenerate_sum_radii:{s:.3e}"
    return s, "ok"


def main():
    # Imitate the sandbox: it prepends `np.random.seed(SEED); random.seed(SEED)`
    # (CirclePacking.preprocess) BEFORE running the program, so seed here the same
    # way to reproduce the sandbox's exact packing and score.
    np.random.seed(SEED)
    random.seed(SEED)
    centers, radii, sum_radii = run_packing()
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float).ravel()

    score, score_msg = sandbox_score(centers, radii)
    print(f"n = {centers.shape[0]}")
    print(f"sum of radii (returned)  = {sum_radii:.6f}")
    print(f"sum of radii (recomputed)= {radii.sum():.6f}")
    print(f"SANDBOX reward (seed {SEED}) = {score:.6f}   [{score_msg}]")
    print(f"radii: min={radii.min():.4f} max={radii.max():.4f} mean={radii.mean():.4f}")

    issues = collect_issues(centers, radii)
    if issues:
        print(f"\nVALIDATION FAILED: {len(issues)} issue(s)")
        for msg in issues[:10]:
            print(f"  - {msg}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("\nValidation OK.")

    save_to = sys.argv[1] if len(sys.argv) > 1 else None
    plot_packing(centers, radii, sum_radii, save_to='out.png')


if __name__ == "__main__":
    main()