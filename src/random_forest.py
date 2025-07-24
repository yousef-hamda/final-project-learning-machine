import numpy as np
import pandas as pd
from collections import Counter
import math
import random
from concurrent.futures import ThreadPoolExecutor
import threading

class RandomForestNode:
    """
    Node structure for Random Forest trees
    Similar to DecisionTreeNode but optimized for ensemble learning
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, samples=0, impurity=0.0):
        self.feature_index = feature_index    # Which feature to split on
        self.threshold = threshold            # Split threshold value
        self.left = left                      # Left child node
        self.right = right                    # Right child node
        self.value = value                    # Predicted class (for leaf nodes)
        self.samples = samples                # Number of samples in this node
        self.impurity = impurity             # Impurity of this node

class RandomForestTree:
    """
    Individual decision tree optimized for Random Forest
    Incorporates feature randomness and bootstrap sampling
    Enhanced version with proper feature importance calculation
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, 
                 max_features=None, criterion='gini', random_state=None):
        """
        Initialize a single tree for the Random Forest
        
        Parameters:
        -----------
        max_depth : int
            Maximum depth of the tree
        min_samples_split : int
            Minimum samples required to split a node
        min_samples_leaf : int
            Minimum samples required in a leaf
        max_features : int, float, str, or None
            Number of features to consider when looking for the best split
        criterion : str
            Split quality measure ('gini' or 'entropy')
        random_state : int
            Random seed for reproducibility
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.root = None
        self.feature_indices = None
        self.n_features = None
        self.feature_importances_ = None
        
        # Set random seed for reproducible results
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
    def _calculate_gini(self, y):
        """Calculate Gini impurity for label array"""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _calculate_entropy(self, y):
        """Calculate entropy for label array"""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))
    
    def _calculate_impurity(self, y):
        """Calculate impurity using specified criterion"""
        if self.criterion == 'gini':
            return self._calculate_gini(y)
        else:
            return self._calculate_entropy(y)
    
    def _calculate_information_gain(self, parent, left_child, right_child):
        """Calculate information gain from a split"""
        n_parent = len(parent)
        n_left = len(left_child)
        n_right = len(right_child)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        parent_impurity = self._calculate_impurity(parent)
        left_weight = n_left / n_parent
        right_weight = n_right / n_parent
        
        weighted_child_impurity = (left_weight * self._calculate_impurity(left_child) + 
                                 right_weight * self._calculate_impurity(right_child))
        
        return parent_impurity - weighted_child_impurity
    
    def _select_random_features(self, n_features):
        """
        Randomly select a subset of features for splitting
        This is the key randomness component in Random Forest
        
        Parameters:
        -----------
        n_features : int
            Total number of features available
            
        Returns:
        --------
        array
            Indices of randomly selected features
        """
        # Determine number of features to select
        if self.max_features is None:
            # Default: square root of total features
            max_features = int(np.sqrt(n_features))
        elif isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                max_features = int(np.log2(n_features))
            else:
                max_features = n_features
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            # Fraction of total features
            max_features = int(self.max_features * n_features)
        else:
            max_features = n_features
        
        # Ensure at least one feature is selected
        max_features = max(1, max_features)
        
        # Randomly select features without replacement
        feature_indices = np.random.choice(n_features, 
                                         size=min(max_features, n_features), 
                                         replace=False)
        return feature_indices
    
    def _find_best_split(self, X, y):
        """
        Find best split considering only a random subset of features
        This implements the feature randomness aspect of Random Forest
        Enhanced version with better split finding
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
            
        Returns:
        --------
        tuple
            (best_feature_index, best_threshold, best_gain)
        """
        best_gain = -1
        best_feature_index = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # Randomly select subset of features to consider
        random_features = self._select_random_features(n_features)
        
        # Only consider the randomly selected features
        for feature_index in random_features:
            feature_values = X[:, feature_index]
            
            # Sort values for efficient threshold selection
            sorted_indices = np.argsort(feature_values)
            sorted_values = feature_values[sorted_indices]
            sorted_labels = y[sorted_indices]
            
            # Skip if all values are the same
            if len(np.unique(sorted_values)) <= 1:
                continue
            
            # Try different thresholds between consecutive unique values
            for i in range(len(sorted_values) - 1):
                if sorted_values[i] == sorted_values[i + 1]:
                    continue
                
                threshold = (sorted_values[i] + sorted_values[i + 1]) / 2
                
                left_y = sorted_labels[:i + 1]
                right_y = sorted_labels[i + 1:]
                
                # Check minimum samples constraint
                if (len(left_y) < self.min_samples_leaf or 
                    len(right_y) < self.min_samples_leaf):
                    continue
                
                gain = self._calculate_information_gain(sorted_labels, left_y, right_y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        
        return best_feature_index, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree with enhanced node information"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Calculate current impurity
        current_impurity = self._calculate_impurity(y)
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1 or
            current_impurity == 0):
            most_common_class = Counter(y).most_common(1)[0][0]
            return RandomForestNode(value=most_common_class, samples=n_samples, impurity=current_impurity)
        
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        if best_feature is None or best_gain <= 0:
            most_common_class = Counter(y).most_common(1)[0][0]
            return RandomForestNode(value=most_common_class, samples=n_samples, impurity=current_impurity)
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            most_common_class = Counter(y).most_common(1)[0][0]
            return RandomForestNode(value=most_common_class, samples=n_samples, impurity=current_impurity)
        
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        return RandomForestNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
            samples=n_samples,
            impurity=current_impurity
        )
    
    def fit(self, X, y):
        """Train the individual tree"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y)
        
        # Calculate feature importances
        self._calculate_feature_importances(X, y)
        
        return self
    
    def _calculate_feature_importances(self, X, y):
        """
        Calculate feature importance based on impurity decrease
        Enhanced version similar to decision tree
        """
        # Initialize importance scores
        importances = np.zeros(self.n_features)
        
        def calculate_importance_recursive(node, n_node_samples):
            """Recursively calculate importance for each feature used in tree"""
            if node.value is not None:  # Leaf node
                return
            
            # Calculate importance for this split
            feature_idx = node.feature_index
            
            # Get left and right child information
            n_left = node.left.samples
            n_right = node.right.samples
            
            # Calculate weighted impurity decrease
            importance = (n_node_samples / X.shape[0]) * (
                node.impurity - 
                (n_left / n_node_samples) * node.left.impurity -
                (n_right / n_node_samples) * node.right.impurity
            )
            
            # Add to feature importance
            importances[feature_idx] += importance
            
            # Recursively calculate for child nodes
            calculate_importance_recursive(node.left, n_left)
            calculate_importance_recursive(node.right, n_right)
        
        # Calculate importance starting from root
        if self.root is not None:
            calculate_importance_recursive(self.root, self.root.samples)
        
        # Normalize importances to sum to 1
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
    
    def _predict_sample(self, x, node):
        """Predict single sample by traversing the tree"""
        if node.value is not None:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Make predictions for multiple samples"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = []
        for x in X:
            prediction = self._predict_sample(x, self.root)
            predictions.append(prediction)
        
        return np.array(predictions)

class RandomForest:
    """
    Random Forest classifier implementation from scratch
    Combines multiple decision trees with bootstrap sampling and feature randomness
    Enhanced version with proper feature importance calculation and out-of-bag scoring
    """
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', criterion='gini',
                 bootstrap=True, oob_score=False, random_state=None, n_jobs=1):
        """
        Initialize Random Forest classifier
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int
            Maximum depth of individual trees
        min_samples_split : int
            Minimum samples required to split a node
        min_samples_leaf : int
            Minimum samples required in a leaf node
        max_features : str, int, float
            Number of features to consider when looking for best split
            - 'sqrt': sqrt(n_features)
            - 'log2': log2(n_features)
            - int: exact number
            - float: fraction of features
        criterion : str
            Split quality criterion ('gini' or 'entropy')
        bootstrap : bool
            Whether to use bootstrap sampling for training each tree
        oob_score : bool
            Whether to calculate out-of-bag score
        random_state : int
            Random seed for reproducibility
        n_jobs : int
            Number of parallel jobs (1 for sequential execution)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.trees = []                       # List to store individual trees
        self.feature_names = None             # Feature names for interpretation
        self.n_features = None                # Number of features
        self.feature_importances_ = None      # Feature importance scores
        self.oob_score_ = None                # Out-of-bag score
        self.oob_indices = []                 # Out-of-bag indices for each tree
        
        # Set random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
    def _bootstrap_sample(self, X, y, random_state=None):
        """
        Create a bootstrap sample from the training data
        Bootstrap sampling involves sampling with replacement
        Enhanced version that also returns out-of-bag indices
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        random_state : int
            Random seed for this specific sample
            
        Returns:
        --------
        tuple
            (X_bootstrap, y_bootstrap, oob_indices) - Bootstrap sample and OOB indices
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = X.shape[0]
        
        if self.bootstrap:
            # Sample with replacement (bootstrap)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            # Out-of-bag indices are those not selected
            oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(indices))
        else:
            # Use all data without replacement
            indices = np.arange(n_samples)
            oob_indices = np.array([])  # No OOB samples when not bootstrapping
        
        return X[indices], y[indices], oob_indices
    
    def _train_single_tree(self, tree_idx, X, y):
        """
        Train a single tree in the forest
        Each tree gets a different bootstrap sample and random seed
        Enhanced version that returns both tree and OOB indices
        
        Parameters:
        -----------
        tree_idx : int
            Index of the tree being trained
        X : array-like
            Training features
        y : array-like
            Training labels
            
        Returns:
        --------
        tuple
            (RandomForestTree, oob_indices) - Trained tree and OOB indices
        """
        # Create unique random state for this tree
        tree_random_state = None
        if self.random_state is not None:
            tree_random_state = self.random_state + tree_idx
        
        # Create bootstrap sample for this tree
        X_bootstrap, y_bootstrap, oob_indices = self._bootstrap_sample(X, y, tree_random_state)
        
        # Create and train the tree
        tree = RandomForestTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            criterion=self.criterion,
            random_state=tree_random_state
        )
        
        tree.fit(X_bootstrap, y_bootstrap)
        return tree, oob_indices
    
    def _calculate_oob_score(self, X, y):
        """
        Calculate out-of-bag score using predictions from trees that didn't see each sample
        This provides an unbiased estimate of model performance
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
        """
        if not self.bootstrap or len(self.oob_indices) == 0:
            return
        
        n_samples = X.shape[0]
        oob_predictions = np.full(n_samples, -1)  # -1 indicates no prediction
        oob_vote_counts = np.zeros((n_samples, len(np.unique(y))))
        classes = np.unique(y)
        
        # For each sample, collect predictions from trees that didn't see it
        for tree_idx, (tree, oob_idx) in enumerate(zip(self.trees, self.oob_indices)):
            if len(oob_idx) > 0:
                # Get predictions for OOB samples
                oob_pred = tree.predict(X[oob_idx])
                
                # Count votes for each class
                for i, sample_idx in enumerate(oob_idx):
                    pred_class = oob_pred[i]
                    class_idx = np.where(classes == pred_class)[0][0]
                    oob_vote_counts[sample_idx, class_idx] += 1
        
        # Make final predictions based on majority vote
        for i in range(n_samples):
            if np.sum(oob_vote_counts[i]) > 0:  # If this sample was OOB for any tree
                oob_predictions[i] = classes[np.argmax(oob_vote_counts[i])]
        
        # Calculate accuracy for samples that have OOB predictions
        valid_predictions = oob_predictions != -1
        if np.sum(valid_predictions) > 0:
            self.oob_score_ = np.mean(oob_predictions[valid_predictions] == y[valid_predictions])
        else:
            self.oob_score_ = None
    
    def fit(self, X, y):
        """
        Train the Random Forest on the given data
        Enhanced version with proper feature importance and OOB scoring
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training labels
            
        Returns:
        --------
        self
            Trained Random Forest object
        """
        # Handle pandas DataFrames
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.n_features = X.shape[1]
        
        print(f"Training Random Forest...")
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Forest: {self.n_estimators} trees")
        print(f"Max features per split: {self.max_features}")
        print(f"Bootstrap sampling: {self.bootstrap}")
        
        self.trees = []
        self.oob_indices = []
        
        if self.n_jobs == 1:
            # Sequential training (easier to debug)
            for i in range(self.n_estimators):
                tree, oob_idx = self._train_single_tree(i, X, y)
                self.trees.append(tree)
                self.oob_indices.append(oob_idx)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"Trained {i + 1}/{self.n_estimators} trees")
        else:
            # Parallel training (faster but more complex)
            print("Training trees in parallel...")
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for i in range(self.n_estimators):
                    future = executor.submit(self._train_single_tree, i, X, y)
                    futures.append(future)
                
                for i, future in enumerate(futures):
                    tree, oob_idx = future.result()
                    self.trees.append(tree)
                    self.oob_indices.append(oob_idx)
                    
                    if (i + 1) % 10 == 0:
                        print(f"Completed {i + 1}/{self.n_estimators} trees")
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        # Calculate out-of-bag score if requested
        if self.oob_score:
            self._calculate_oob_score(X, y)
            if self.oob_score_ is not None:
                print(f"Out-of-bag score: {self.oob_score_:.4f}")
        
        print("✅ Random Forest training completed!")
        return self
    
    def _calculate_feature_importances(self):
        """
        Calculate feature importance as the average importance across all trees
        Enhanced version using proper impurity-based importance
        """
        if not self.trees or self.n_features is None:
            return
        
        # Average feature importances across all trees
        total_importances = np.zeros(self.n_features)
        
        for tree in self.trees:
            if hasattr(tree, 'feature_importances_') and tree.feature_importances_ is not None:
                total_importances += tree.feature_importances_
        
        # Normalize to get average importance
        if np.sum(total_importances) > 0:
            self.feature_importances_ = total_importances / len(self.trees)
        else:
            self.feature_importances_ = total_importances
    
    def predict(self, X):
        """
        Make predictions using majority voting from all trees
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Features to make predictions for
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted class labels
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Collect predictions from all trees
        tree_predictions = []
        for tree in self.trees:
            predictions = tree.predict(X)
            tree_predictions.append(predictions)
        
        # Convert to matrix: rows = samples, columns = trees
        tree_predictions = np.array(tree_predictions).T
        
        # Majority voting for each sample
        final_predictions = []
        for sample_predictions in tree_predictions:
            # Find most common prediction across all trees
            most_common = Counter(sample_predictions).most_common(1)[0][0]
            final_predictions.append(most_common)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for each sample
        Probabilities are computed as the fraction of trees voting for each class
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Features to predict probabilities for
            
        Returns:
        --------
        tuple
            (probabilities, classes) where probabilities is array of shape (n_samples, n_classes)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Collect predictions from all trees
        tree_predictions = []
        for tree in self.trees:
            predictions = tree.predict(X)
            tree_predictions.append(predictions)
        
        tree_predictions = np.array(tree_predictions).T
        
        # Find all unique classes
        unique_classes = np.unique(np.concatenate(tree_predictions))
        
        # Calculate probabilities for each sample
        probabilities = []
        for sample_predictions in tree_predictions:
            class_counts = Counter(sample_predictions)
            sample_proba = []
            for cls in unique_classes:
                # Probability = fraction of trees voting for this class
                prob = class_counts.get(cls, 0) / len(sample_predictions)
                sample_proba.append(prob)
            probabilities.append(sample_proba)
        
        return np.array(probabilities), unique_classes
    
    def feature_importance(self):
        """
        Get feature importance scores
        
        Returns:
        --------
        array-like
            Feature importance scores (normalized to sum to 1)
        """
        return self.feature_importances_
    
    def get_params(self):
        """Get hyperparameters of the Random Forest"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'criterion': self.criterion,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'random_state': self.random_state
        }

# Example usage and testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import time
    
    print("Testing Random Forest Implementation")
    print("=" * 50)
    
    # Create sample dataset
    print("Creating sample dataset...")
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=7, 
                             n_redundant=2, n_classes=3, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    start_time = time.time()
    
    rf = RandomForest(n_estimators=50, max_depth=8, max_features='sqrt', 
                     criterion='gini', bootstrap=True, oob_score=True, 
                     random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = rf.predict(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nRandom Forest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate probabilities
    probabilities, classes = rf.predict_proba(X_test[:5])
    print(f"\nPrediction probabilities for first 5 samples:")
    for i, (proba, true_label, pred_label) in enumerate(zip(probabilities, y_test[:5], y_pred[:5])):
        print(f"Sample {i+1}: Predicted={pred_label}, True={true_label}")
        for j, cls in enumerate(classes):
            print(f"  Class {cls}: {proba[j]:.3f}")
    
    # Feature importance
    importances = rf.feature_importance()
    if importances is not None:
        print(f"\nFeature Importance:")
        for i, importance in enumerate(importances):
            print(f"Feature {i}: {importance:.4f}")
    
    print(f"\nModel Parameters:")
    params = rf.get_params()
    for key, value in params.items():
        print(f"{key}: {value}")
    
    print("\n✅ Random Forest test completed successfully!")