# Bonus Feature: Least Squares Multi-Line Intersection (+1 point)

## Overview
This bonus feature computes the intersection point of **multiple lines** (more than 2) using **least squares fitting** and estimates the **uncertainty** of the intersection.

## Mathematical Foundation

### Problem Setup
For each detected line, we have the slope-intercept form:
$$y = kx + b$$

Rewriting in standard form:
$$kx - y + b = 0$$

### Matrix Formulation
For N lines, we construct a system:
$$\mathbf{A} \mathbf{p} = \mathbf{b}$$

Where:
- $\mathbf{A}$ is an N×2 matrix: each row is $[k_i, -1]$
- $\mathbf{p} = [x, y]^T$ is the intersection point we're solving for
- $\mathbf{b}$ is a vector of $[-b_i]$ values

### Least Squares Solution
When lines don't intersect at a single point (due to noise), we minimize:
$$\min_{\mathbf{p}} \|\mathbf{A}\mathbf{p} - \mathbf{b}\|^2$$

The solution is:
$$\mathbf{p} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$$

This finds the optimal point that best fits all lines simultaneously.

## Uncertainty Estimation

The uncertainty is estimated from the **residual error**:
$$\text{MSE} = \frac{\sum_{i=1}^{N} \text{residual}_i^2}{N}$$

$$\text{Uncertainty} = \sqrt{\text{MSE}}$$

This represents the standard deviation (in pixels) of the intersection estimate. A smaller value indicates higher confidence.

## Code Implementation

### New Function: `compute_least_squares_intersection()`
```python
def compute_least_squares_intersection(lines_slopes_intercepts):
    """
    Computes intersection of multiple lines using least squares
    Returns: (x, y, uncertainty)
    """
    # Build matrix A and vector b
    A = []
    b_vec = []
    for k, b in lines_slopes_intercepts:
        A.append([k, -1])
        b_vec.append(-b)
    
    # Solve using numpy's lstsq
    p, residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)
    x, y = p[0], p[1]
    
    # Calculate uncertainty from residuals
    uncertainty = np.sqrt(residuals[0] / len(lines_slopes_intercepts))
    
    return x, y, uncertainty
```

## Visualization

In the output image:
- **Red circles**: Pairwise intersections (original method)
- **Yellow circle with radius**: Least squares intersection with uncertainty radius
- **Text label**: Shows coordinates and uncertainty estimate

Example output:
```
Least Squares Intersection: (256.34, 412.56)
Uncertainty (Std Dev): ±2.1543 pixels
Number of lines used: 4
```

## Advantages Over Pairwise Method

| Aspect | Pairwise | Least Squares |
|--------|----------|---------------|
| Lines used | 2 at a time | All lines simultaneously |
| Robustness | Affected by outliers | More robust to noise |
| Uncertainty | Not quantified | Quantified via residuals |
| Accuracy | Lower with many lines | Higher with redundancy |

## Example Scenario
- **Detected 4 lines** representing field markings
- **Pairwise method**: 6 different intersection points (C(4,2) = 6)
- **Least squares**: 1 optimal intersection point with uncertainty estimate

## How to Use
The bonus feature automatically runs when 2 or more lines are detected. The results are printed to console and visualized in the output window.

Results show:
- Intersection coordinates (x, y)
- Uncertainty in pixels (±value)
- Number of lines contributing to the estimate
