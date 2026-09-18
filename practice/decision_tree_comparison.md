# Comprehensive Comparison: Custom `DecisionTreeRegressor` vs. `scikit-learn`

This document presents an in-depth technical analysis comparing a custom, from-scratch Python/NumPy implementation of a **Decision Tree Regressor** against the production-grade **`sklearn.tree.DecisionTreeRegressor`**.

---

## Table of Contents
1. [Architectural Overview](#1-architectural-overview)
2. [Splitting Algorithm & Mathematical Mechanics](#2-splitting-algorithm--mathematical-mechanics)
3. [Time & Space Complexity Analysis](#3-time--space-complexity-analysis)
4. [Data Structures & Memory Layout](#4-data-structures--memory-layout)
5. [Inference & Traversal Mechanics](#5-inference--traversal-mechanics)
6. [Regularization, Stopping Criteria & Pruning](#6-regularization-stopping-criteria--pruning)
7. [Feature Handling & Edge Cases](#7-feature-handling--edge-cases)
8. [Empirical Benchmark & Metrics](#8-empirical-benchmark--metrics)
9. [Detailed Comparison Matrix](#9-detailed-comparison-matrix)
10. [Step-by-Step Optimization Roadmap for Custom Trees](#10-step-by-step-optimization-roadmap-for-custom-trees)

---

## 1. Architectural Overview

```
+------------------------------------------------------------------------------------------+
|                                 HIGH-LEVEL ARCHITECTURE                                  |
+------------------------------------------------------------------------------------------+
| Feature                     Custom Implementation          scikit-learn (CART Engine)    |
| --------------------------  -----------------------------  ----------------------------- |
| Core Language               Pure Python + NumPy            Cython / C Extensions         |
| Split Finding Strategy      Brute-force boolean masking    Presorted incremental scan    |
| Memory Layout               Object graph of `Node` heap    Contiguous 1D C-struct arrays |
| Metric Evaluation           Recomputed slice MSE           Running sum updates (O(1))    |
| Tree Traversal              Recursive DFS per sample       Iterative flat array indexing |
| Pruning                     None (Pre-stopping only)       Cost-Complexity Pruning (CCP) |
+------------------------------------------------------------------------------------------+
```

---

## 2. Splitting Algorithm & Mathematical Mechanics

Both implementations minimize the **Sum of Squared Residuals (SSR)** or **Mean Squared Error (MSE)** to find optimal split thresholds $\theta$ across candidate features $j$.

$$\text{MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y}_S)^2$$

Where the goal at each node is to maximize the impurity decrease (variance reduction):

$$\Delta I = \text{MSE}(S) - \left( \frac{|S_L|}{|S|} \text{MSE}(S_L) + \frac{|S_R|}{|S|} \text{MSE}(S_R) \right)$$

### A. Custom Implementation (Brute-Force Slice Evaluation)
In the custom implementation, candidate splits are evaluated by explicitly constructing boolean masks on arrays and calculating the statistics from scratch:

```python
for threshold in thresholds:
    left_mask = feature_values <= threshold
    right_mask = feature_values > threshold

    y_left, y_right = y[left_mask], y[right_mask]

    # Recomputes mean and SSR across all elements in every iteration
    ssr_left = np.sum((y_left - np.mean(y_left)) ** 2)
    ssr_right = np.sum((y_right - np.mean(y_right)) ** 2)
    total_mse = (ssr_left + ssr_right) / num_samples
```
- **Computational Bottleneck:** For $N$ samples at a node, there are up to $N - 1$ distinct thresholds.
- Calculating `left_mask`, `np.mean()`, and `np.sum(...)` involves scanning $N$ items **for each threshold**.
- Evaluating one feature with $N$ samples costs $\mathcal{O}(N \times N) = \mathcal{O}(N^2)$ operations. Across $M$ features, each node split costs **$\mathcal{O}(M \cdot N^2)$**.

---

### B. `scikit-learn` (`_splitter.pyx` & `_criterion.pyx`)
Scikit-learn utilizes an **incremental single-pass sliding window** algorithm implemented in Cython:

1. **Sort Feature Values Once:**
   Samples are sorted by the feature values in $\mathcal{O}(N \log N)$:
   $$x_{(1)} \le x_{(2)} \le \dots \le x_{(N)}$$
2. **Maintain Running Accumulators:**
   Using the algebraic expansion of sum of squares:
   $$\text{SSR} = \sum_{i=1}^{k} (y_i - \bar{y})^2 = \sum_{i=1}^{k} y_i^2 - \frac{1}{k}\left(\sum_{i=1}^{k} y_i\right)^2$$
   The splitter maintains four scalar accumulators:
   - $W_L = \sum_{i \in L} w_i$, $\quad W_R = \sum_{i \in R} w_i$
   - $S_L = \sum_{i \in L} y_i$, $\quad S_R = \sum_{i \in R} y_i$
   - $SS_L = \sum_{i \in L} y_i^2$, $\quad SS_R = \sum_{i \in R} y_i^2$
3. **$\mathcal{O}(1)$ Boundary Step:**
   Moving the split threshold from sample $k$ to $k+1$ requires transferring one sample $y_{k+1}$ from the right partition to the left partition:
   $$S_L \leftarrow S_L + y_{k+1}, \quad S_R \leftarrow S_R - y_{k+1}$$
   $$SS_L \leftarrow SS_L + y_{k+1}^2, \quad SS_R \leftarrow SS_R - y_{k+1}^2$$
   The resulting MSE is updated in **$\mathcal{O}(1)$ constant time** without touching the rest of the array.
4. **Complexity:** $\mathcal{O}(N \log N)$ for sorting + $\mathcal{O}(N)$ for the linear scan = **$\mathcal{O}(N \log N)$** per feature, executed directly in native C.

---

## 3. Time & Space Complexity Analysis

| Metric | Custom Implementation | scikit-learn (`CART`) | Ratio / Speedup |
| :--- | :--- | :--- | :--- |
| **Node Split Time** | $\mathcal{O}(M \cdot N^2)$ | $\mathcal{O}(M \cdot N \log N)$ | $\approx \frac{N}{\log N} \times$ (Cython adds another $50\times$) |
| **Full Tree Build (Balanced, depth $D$)** | $\mathcal{O}(M \cdot D \cdot N^2)$ | $\mathcal{O}(M \cdot D \cdot N \log N)$ | Dramatic on $N > 1,000$ |
| **Prediction Time ($K$ samples)** | $\mathcal{O}(K \cdot D)$ (Python call stack) | $\mathcal{O}(K \cdot D)$ (Contiguous C loop) | $10\times - 100\times$ faster in sklearn |
| **Memory per Node** | $\ge 152\text{ bytes}$ (Python Heap) | $32 - 48\text{ bytes}$ (Aligned C struct) | $\approx 4\times$ memory reduction |
| **Cache Locality** | Poor (Scattered heap references) | Optimal (Contiguous 1D buffers) | Enables CPU L1/L2 prefetching |

---

## 4. Data Structures & Memory Layout

```
CUSTOM IMPLEMENTATION (Object Graph)
+----------------+      +----------------+
|  Node (Root)   | ---> |   Node (Left)  | ---> Leaf Node
|  - feature: 2  |      |   - feature: 0 |
|  - thresh: 4.1 |      |   - value: None|
|  - left: ptr   |      +----------------+
|  - right: ptr  | ---> +----------------+
+----------------+      |   Node (Right) | ---> Leaf Node
                        +----------------+
* High Python object overhead (PyObject_HEAD + __dict__)
* Memory fragmented across Python heap

SKLEARN IMPLEMENTATION (Contiguous Flat Buffers in C)
Index:              0     1     2     3     4     5
children_left:    [ 1,    3,   -1,   -1,   -1,   -1 ]  (-1 indicates leaf)
children_right:   [ 2,    4,   -1,   -1,   -1,   -1 ]
feature:          [ 2,    0,   -2,   -2,   -2,   -2 ]  (-2 indicates leaf)
threshold:        [4.1,  1.8,  -2.0, -2.0, -2.0, -2.0]
value:            [0.0,  0.0,  12.4, 18.2, 25.1, 30.5]
* Extremely compact, zero pointer dereferencing, L1/L2 cache prefetching
```

### The Custom Node Class
Each node instance incurs:
- `PyObject` header: 16 bytes.
- Type pointer: 8 bytes.
- Instance dictionary (`__dict__`): 104+ bytes.
- Attribute references (`feature`, `threshold`, `left`, `right`, `value`): 40+ bytes.
- **Total:** $>152\text{ bytes}$ per node, allocated dynamically across disjoint heap locations.

### Scikit-Learn `Tree` Struct
Defined in Cython (`sklearn.tree._tree.Tree`), a node is represented as an aligned C struct:

```c
typedef struct {
    intp_t left_child;       // Index of left child
    intp_t right_child;      // Index of right child
    intp_t feature;          // Feature index for split
    float64_t threshold;     // Threshold value
    float64_t impurity;      // Node MSE / impurity
    intp_t n_node_samples;   // Sample count reaching node
    float64_t weighted_n_node_samples;
    uint8_t missing_go_to_left;
} Node;
```

All nodes are stored in contiguous 1D arrays, ensuring instant sequential reads and vectorization.

---

## 5. Inference & Traversal Mechanics

### Custom Prediction:
```python
def predict(self, X):
    return np.array([self._traverse_tree(x, self.tree) for x in X])

def _traverse_tree(self, x, node):
    if node.is_leaf_node():
        return node.value
    if x[node.feature] <= node.threshold:
        return self._traverse_tree(x, node.left)
    return self._traverse_tree(x, node.right)
```
- Incurs Python interpreter dispatch overhead and creates a new Python call frame for every node in the branch for each query row.
- Total function calls: $\mathcal{O}(N_{\text{test}} \times \text{depth})$.

### Scikit-Learn Prediction:
- Traversal is compiled directly in C (`_tree.pyx:predict`).
- It runs an iterative `while` loop that directly indexes the integer array:
  ```c
  while node.left_child != _TREE_LEAF:
      if X[i, node.feature] <= node.threshold:
          node_id = node.left_child
      else:
          node_id = node.right_child
      node = &nodes[node_id]
  out[i] = node.value
  ```
- **Zero function call overhead**, zero heap allocation during prediction.

---

## 6. Regularization, Stopping Criteria & Pruning

| Hyperparameter / Mechanism | Custom Implementation | scikit-learn | Impact on Generalization |
| :--- | :---: | :---: | :--- |
| `max_depth` | Supported | Supported | Prevents deep, memorizing trees |
| `min_samples_split` | Supported | Supported | Halts split if sample size too small |
| `min_samples_leaf` | Missing | Supported | Ensures leaf nodes have statistical mass |
| `min_impurity_decrease` | Missing | Supported | Rejects splits with negligible gain |
| `max_leaf_nodes` | Missing | Supported | Grows tree in best-first priority |
| `max_features` | Missing | Supported | Selects random feature subsets (RF basis) |
| **Cost-Complexity Pruning (`ccp_alpha`)** | Missing | Supported | Post-pruning via Weakest Link Algorithm |

### Minimal Cost-Complexity Pruning ($R_\alpha(T)$)
Scikit-Learn implements Breiman's weakest-link pruning:
$$R_\alpha(T) = R(T) + \alpha |T|$$
Where $R(T)$ is total training error and $|T|$ is the number of terminal leaves. Setting `ccp_alpha` prunes subtrees that fail to justify their complexity, producing smaller, more robust decision trees than pure pre-stopping.

---

## 7. Feature Handling & Edge Cases

1. **Missing Values (NaNs):**
   - *Custom:* Raises errors or produces invalid boolean comparisons.
   - *Scikit-Learn:* Fully supports missing values natively (`missing_go_to_left`), evaluating whether missing values yield higher gain going left or right.
2. **Categorical Variables:**
   - Neither implementation natively splits unstructured categorical strings without one-hot or ordinal encoding, but Scikit-Learn provides strict validation via `check_array`.
3. **Multi-Target Regression ($Y \in \mathbb{R}^{N \times K}$):**
   - *Custom:* Strictly single-output ($Y \in \mathbb{R}^N$).
   - *Scikit-Learn:* Supports vector targets out-of-the-box by computing covariance/variance across multiple target columns simultaneously.

---

## 8. Empirical Benchmark & Metrics

Benchmarked on **Diabetes / California Housing Benchmark** with `max_depth=5` and `min_samples_split=5`:

```
+------------------------------------+----------------+--------------------+-----------+-----------+----------+
| Model                              | Train Time (s) | Inference Time (s) | MSE       | RMSE      | R² Score |
+------------------------------------+----------------+--------------------+-----------+-----------+----------+
| Custom DecisionTreeRegressor       | 0.1407 s       | 0.0035 s           | 3692.19   | 60.76     | 0.3031   |
| scikit-learn DecisionTreeRegressor | 0.0030 s       | 0.0003 s           | 3740.04   | 61.16     | 0.2941   |
+------------------------------------+----------------+--------------------+-----------+-----------+----------+
```

### Key Empirical Findings:
1. **Predictive Equivalence:** Both models achieve nearly identical validation performance ($R^2 \approx 0.30$), confirming the mathematical correctness of your custom tree's splitting logic.
2. **Training Speed Difference:** Scikit-learn trains **$\approx 47\times$ faster** due to Cython compilation and avoiding repetitive array masking.
3. **Inference Latency:** Scikit-learn predicts **$\approx 10\times$ faster** by substituting Python recursive calls with C pointer updates.

---

## 9. Detailed Comparison Matrix

| Dimension | Custom Implementation | scikit-learn `DecisionTreeRegressor` |
| :--- | :--- | :--- |
| **Paradigm** | Educational / Interpretable Prototype | High-Performance Production Library |
| **Dependencies** | Pure Python, `numpy`, `pandas` | Cython, C, NumPy, SciPy |
| **Tree Storage** | Object-Oriented Graph of `Node` instances | Compact 1D structured NumPy array buffers |
| **Split Finding** | Brute-force array masking $\mathcal{O}(M \cdot N^2)$ | Incremental sliding window $\mathcal{O}(M \cdot N \log N)$ |
| **Pruning Methods** | Early stopping (`max_depth`, `min_samples_split`) | Pre-pruning + Cost-Complexity Post-Pruning (`ccp_alpha`) |
| **Missing Values** | Not supported (requires explicit preprocessing) | Natively supported (`missing_go_to_left` routing) |
| **Sample Weights** | Not supported | Supported via `sample_weight` array |
| **Multi-Output** | Single target variable only | Supports multi-dimensional $Y$ targets |
| **Export Formats** | Manual string / dict representation | Graphviz DOT (`export_graphviz`), text rules (`export_text`) |

---

## 10. Step-by-Step Optimization Roadmap for Custom Trees

To scale your custom tree towards production performance without compiling C/Cython:

1. **Adopt Incremental Variance Calculation:**
   Replace the inner threshold loop with pre-sorted cumulative sums:
   ```python
   # Sort once per feature:
   order = np.argsort(X_col)
   y_sorted = y[order]
   
   # Prefix sums:
   sum_l = np.cumsum(y_sorted)[:-1]
   sum_r = sum_l[-1] - sum_l
   ```
2. **Quantile / Histogram Binning:**
   Instead of testing every value, select 32–64 quantiles via `np.percentile`:
   ```python
   thresholds = np.percentile(feature_values, np.linspace(1, 99, 32))
   ```
3. **Vectorize Predictions:**
   Flatten tree queries using vectorized condition matrices or convert the `Node` graph into flat parallel arrays for zero-overhead evaluation.
