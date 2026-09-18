import numpy as np

# -----------------------------------------------------------------------------
# 1. Base Components (Decision Tree)
# -----------------------------------------------------------------------------

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, gain=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gain = gain
        
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, task='regression', criterion=None, max_depth=10, min_samples_split=2, 
                 min_samples_leaf=1, max_features=None):
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
        
        if criterion is None:
            self.criterion = 'mse' if task == 'regression' else 'gini'
        else:
            self.criterion = criterion
            
    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y)
        return self
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y)) if self.task == 'classification' else -1

        # Stopping criteria
        if (depth >= self.max_depth or n_samples < self.min_samples_split or
            (self.task == 'classification' and n_labels <= 1) or 
            (self.task == 'regression' and np.var(y) == 0)):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Feature subsampling
        feat_idxs = self._get_feature_indices(n_feats)
        
        # Greedy search
        best_feature, best_thresh, best_gain = self._best_split(X, y, feat_idxs)
        
        # If no gain, make a leaf
        if best_gain <= 1e-7:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)
            
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)
            
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(feature=best_feature, threshold=best_thresh, left=left, right=right, gain=best_gain)
        
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            # Simple optimization: only check percentiles to speed up
            if len(thresholds) > 100:
                thresholds = np.percentile(thresholds, np.linspace(1, 99, 100))
                
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                left_idxs, right_idxs = self._split(X_column, thr)
                
                if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
                    continue
                    
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
                    
        return split_idx, split_threshold, best_gain

    def _information_gain(self, y, X_column, threshold):
        # Calculate parent impurity/variance
        parent_impurity = self._calculate_impurity(y)
        
        # Generate split
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
            
        # Calculate weighted child impurity/variance
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._calculate_impurity(y[left_idxs]), self._calculate_impurity(y[right_idxs])
        child_impurity = (n_l / n) * e_l + (n_r / n) * e_r
        
        gain = parent_impurity - child_impurity
        return gain

    def _calculate_impurity(self, y):
        if self.task == 'regression':
            # Variance for MSE
            return np.var(y)
        else:
            # Classification
            if len(y) == 0:
                return 0
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            
            if self.criterion == 'gini':
                return 1.0 - np.sum(probabilities ** 2)
            elif self.criterion == 'entropy':
                return -np.sum(probabilities * np.log2(probabilities + 1e-9))
                
        return 0

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _calculate_leaf_value(self, y):
        if self.task == 'regression':
            return np.mean(y)
        else:
            labels, counts = np.unique(y, return_counts=True)
            return labels[np.argmax(counts)]

    def _get_feature_indices(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)
        elif self.max_features == 'sqrt':
            max_f = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_f = int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            max_f = int(self.max_features * n_features)
        else:
            max_f = n_features
        return np.random.choice(n_features, max_f, replace=False)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
            
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# -----------------------------------------------------------------------------
# 2. Random Forest Regressor
# -----------------------------------------------------------------------------

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, 
                 max_features='sqrt', bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTree(
                task='regression',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            X_samp, y_samp = self._bootstrap_samples(X, y) if self.bootstrap else (X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)
        return self
            
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
        
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Average over all trees
        return np.mean(tree_preds, axis=0)

# -----------------------------------------------------------------------------
# 3. Gradient Boosting Regressor (MSE Loss)
# -----------------------------------------------------------------------------

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.F0 = None
        
    def fit(self, X, y):
        # Step 1: Initialize model with constant value (mean of y)
        self.F0 = np.mean(y)
        Fm = np.full(len(y), self.F0)
        
        self.trees = []
        for m in range(self.n_estimators):
            # Step 2(a): Compute pseudo-residuals (negative gradient for MSE)
            rm = y - Fm
            
            # Step 2(b): Fit a regression tree to the residuals
            tree = DecisionTree(
                task='regression',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, rm)
            self.trees.append(tree)
            
            # Step 2(c): Update the model
            update = tree.predict(X)
            Fm += self.learning_rate * update
            
        return self

    def predict(self, X):
        Fm = np.full(X.shape[0], self.F0)
        for tree in self.trees:
            Fm += self.learning_rate * tree.predict(X)
        return Fm

# -----------------------------------------------------------------------------
# 4. XGBoost Regressor (Exact Greedy, G & H, L2 Reg)
# -----------------------------------------------------------------------------

class XGBoostNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, weight=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.weight = weight # Optimal leaf weight (w*)
        
    def is_leaf_node(self):
        return self.weight is not None

class XGBoostTree:
    def __init__(self, max_depth=3, min_samples_split=2, reg_lambda=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.root = None
        
    def fit(self, X, g, h):
        self.root = self._grow_tree(X, g, h)
        return self
        
    def _grow_tree(self, X, g, h, depth=0):
        n_samples = X.shape[0]
        
        # Calculate node optimal weight
        G = np.sum(g)
        H = np.sum(h)
        weight = -G / (H + self.reg_lambda)
        
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return XGBoostNode(weight=weight)
            
        best_gain = 0
        best_feature, best_thresh = None, None
        
        n_features = X.shape[1]
        
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            if len(thresholds) > 100:
                thresholds = np.percentile(thresholds, np.linspace(1, 99, 100))
                
            for thr in thresholds:
                left_idxs = X_column <= thr
                right_idxs = X_column > thr
                
                if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
                    continue
                    
                G_L, H_L = np.sum(g[left_idxs]), np.sum(h[left_idxs])
                G_R, H_R = np.sum(g[right_idxs]), np.sum(h[right_idxs])
                
                # Split gain formula
                gain = 0.5 * ((G_L**2 / (H_L + self.reg_lambda)) + 
                              (G_R**2 / (H_R + self.reg_lambda)) - 
                              (G**2 / (H + self.reg_lambda))) - self.gamma
                              
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_thresh = thr
                    
        if best_gain <= 0:
            return XGBoostNode(weight=weight)
            
        # Perform split
        left_idxs = X[:, best_feature] <= best_thresh
        right_idxs = X[:, best_feature] > best_thresh
        
        left = self._grow_tree(X[left_idxs, :], g[left_idxs], h[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], g[right_idxs], h[right_idxs], depth + 1)
        
        return XGBoostNode(feature=best_feature, threshold=best_thresh, left=left, right=right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.weight
            
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class XGBoostRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 reg_lambda=1.0, gamma=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.trees = []
        self.base_score = None
        
    def fit(self, X, y):
        # Initialize with mean
        self.base_score = np.mean(y)
        y_pred = np.full(len(y), self.base_score)
        
        self.trees = []
        for _ in range(self.n_estimators):
            # For MSE loss: L = 0.5 * (y - y_pred)^2
            # gradient g = dL/dy_pred = y_pred - y
            # hessian h = d^2L/dy_pred^2 = 1.0
            g = y_pred - y
            h = np.ones_like(y)
            
            tree = XGBoostTree(
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma
            )
            tree.fit(X, g, h)
            self.trees.append(tree)
            
            # Update predictions
            y_pred += self.learning_rate * tree.predict(X)
            
        return self

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_score)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


# -----------------------------------------------------------------------------
# 5. Demonstration Block
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    
    # 1. Generate synthetic non-linear regression data
    np.random.seed(42)
    X = np.random.rand(200, 1) * 10
    # Function: y = sin(X) + linear trend + noise
    y = np.sin(X).ravel() + 0.1 * X.ravel() + np.random.randn(200) * 0.2
    
    # Train-test split (80-20)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(y))
    
    X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
        
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    print("=== Custom Regression Models Evaluation ===\n")
    
    models = {
        "Decision Tree Regressor": DecisionTree(task='regression', max_depth=5),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=30, max_depth=5, max_features=1.0),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3),
        "XGBoost Regressor": XGBoostRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, reg_lambda=1.0, gamma=0.1)
    }
    
    for name, model in models.items():
        start_t = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_t
        
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        test_mse = mse(y_test, preds)
        
        print(f"[{name}]")
        print(f" - Train Time : {train_time:.4f} sec")
        print(f" - Test R2    : {r2:.4f}")
        print(f" - Test MSE   : {test_mse:.4f}\n")
        
    # Example of DecisionTreeClassifier with Gini
    print("=== Testing Decision Tree Classifier (Gini) ===")
    y_clf = (y > np.median(y)).astype(int)
    X_train_c, X_test_c = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train_c, y_test_c = y_clf[indices[:split_idx]], y_clf[indices[split_idx:]]
    
    clf = DecisionTree(task='classification', criterion='gini', max_depth=3)
    clf.fit(X_train_c, y_train_c)
    preds_c = clf.predict(X_test_c)
    acc = np.mean(preds_c == y_test_c)
    print(f" - Test Accuracy: {acc:.4f}\n")
