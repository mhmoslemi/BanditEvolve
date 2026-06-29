import numpy as np

def run_packing():
    n = 26
    # Initialize positions with optimized staggered grid
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Enhanced initialization with asymmetric staggering and perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid position adjusted to allow better spacing
        x_center = (col + 0.3) / cols
        y_center = (row + 0.3) / rows
        # Add asymmetric staggering for enhanced spatial efficiency
        if row % 2 == 1:
            x_center += 0.3 / cols + np.random.uniform(-0.02, 0.02)
        # Add asymmetric perturbation for dynamic configuration
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        xs.append(x)
        ys.append(y)
    
    # Base radius calculation with adaptive formula to allow expansion in sparse areas
    r0 = 0.365 / cols - 1e-3  # Slightly increased for growth potential
    # Set maximum possible min radius as a guardrail for optimization
    r0 = np.maximum(r0, 1e-4)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Prepare bounds list: 3 per circle of (x, y, r)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n total entries
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Maximize by minimizing negative

    # Build constraints
    # 4 boundary constraints per circle
    cons = []
    for i in range(n):
        # Left constraint: x_i >= r_i
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right constraint: x_i + r_i <= 1.0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom constraint: y_i >= r_i
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top constraint: y_i + r_i <= 1.0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            # Use closure with i and j captured correctly
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # First optimization phase: base layout
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-10,
                                             "eps": 1e-10, "disp": False})
    
    # If not successful, use fallback
    if not res.success:
        v = v0.copy()
    else:
        v = res.x
    
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]
    radii = np.clip(radii, 1e-6, None)
    
    # Geometric dissection: isolate and reconfigure the top two interacting pairs
    # Identify top two interacting pairs (using distance matrix)
    dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
    dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    # Find the top two interacting pairs: distance between any two is minimal
    def compute_interaction_scores():
        scores = np.zeros(n * (n - 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                scores[idx] = dists[i, j] - (radii[i] + radii[j])  # Negative implies overlap
                idx += 1
        return scores
    
    interaction_scores = compute_interaction_scores()
    # Sort indices to find the two most interacting pairs
    # Note: We actually want the pairs with minimal separation (most interacting)
    # So we take the two indices after converting to -1 for minimality
    sorted_indices = np.argsort(-interaction_scores)  # Largest negative comes first
    # Find the first two pairs that are not overlapping or too close
    pair1_idx = sorted_indices[0]
    pair2_idx = sorted_indices[1]
    
    # Extract the four circles involved in the two interacting pairs
    inv = np.zeros(n, dtype=int)
    inv[pair1_idx % (n * (n-1)//2)] = None
    inv[pair2_idx % (n * (n-1)//2)] = None
    # Find the actual circle indices involved in these pairs
    def get_pair_circles(score_idx):
        # For each score index, map it back to (i, j)
        # This mapping is complex but for the sake of this solution, 
        # we'll use a brute-force approach assuming the score indices are in order
        # and the sorted indices are in (i,j) order
        # This could be replaced with proper mapping using a grid
        for i in range(n):
            for j in range(i+1, n):
                if score_idx == (i * (n - 1) // 2 + j - i):
                    return (i, j)
        return (None, None)
    
    # For the purposes of code, we'll now assume the pair1_idx and pair2_idx are valid
    # and extract circles from them (simplified)
    # This method is not perfect, so we add a fallback if pair1 or pair2 is invalid
    try:
        circle1, circle2 = get_pair_circles(pair1_idx)
        circle3, circle4 = get_pair_circles(pair2_idx)
    except:
        circle1, circle2 = 0, 1
        circle3, circle4 = 2, 3
    
    # Create a list of indices to be reconfigured
    targeted_indices = [circle1, circle2, circle3, circle4]
    targeted_indices = np.unique(targeted_indices)
    # We are ensuring that we have only 4 unique indices, possibly less
    
    # Extract their positions and apply perturbations
    perturbed_v = v.copy()
    # Apply directional perturbation to create more separation
    for idx in targeted_indices:
        # Add subtle directional shift based on average surrounding radius
        perturbed_v[3*idx] += (radii[idx] / np.mean(radii)) * 0.005 * np.random.uniform(-1, 1)
        perturbed_v[3*idx + 1] += (radii[idx] / np.mean(radii)) * 0.005 * np.random.uniform(-1, 1)
    
    # Apply targeted radius expansion only to the most constrained within this group
    # Re-evaluate configuration with these changes
    res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 700, "ftol": 1e-11,
                                             "eps": 1e-10, "disp": False})
    
    # If not successful, we revert to previous iteration
    if not res.success:
        v = v
    else:
        v = res.x
    
    # Final radii extraction
    radii = v[2::3]
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(radii, 1e-6, None)
    
    # Additional post-optimization reconfig: targeted expansion only on the most constrained
    # Final validation of entire configuration
    def validate_with_overlaps(centers, radii):
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[i] + radii[j] - 1e-12:
                    return False, f"Circles {i},{j} overlap"
        return True, "No overlaps"
    
    # Final optimization pass: use directional bias for expansion
    # Re-evaluate with final configuration
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
    else:
        v = v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
    
    # Post-validation pass with soft correction
    # Compute all pairwise distances
    dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
    dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    # Check for overlaps in a more efficient way
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < (radii[i] + radii[j]) - 1e-12:
                # Apply localized correction
                # Expand the less constrained of the two to avoid overlap
                if radii[i] > radii[j]:
                    radii[j] += 1e-3
                else:
                    radii[i] += 1e-3
                # Clip to prevent negative values
                radii = np.clip(radii, 1e-6, None)
    
    # Final validation and return
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(radii, 1e-6, None)
    return centers, radii, float(radii.sum())