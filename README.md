# PyLines

Unboxing the black box that is Machine Learning. 

## About

Most managers and developers don't trust machine learning today for various reasons. One of the biggest reasons is that they consider Machine Learning a blackbox that does something weird and can never be comprehended by humans. This repository uses the most basic libraries and implements Machine Learning Algorithms. Each notebook is a separate Algorithm. We simply use a sample dataset to establish how bad our most basic implementation is as compared to scikit learn. 

## Table of Contents:

 - [Algorithms](#algorithms)
     * [Linear Regression](#linear-regression)
     * [Logistic Regression](#logistic-regression)
     * [Decision Tree Regressor](#decision-tree-regressor)
     * [Random Forests Regressor](#random-forest-regressor)
 - [Kaggle](#kaggle)
     * [Blue book for Bulldozers](#blue-book-for-bulldozers)

## Algorithms

### Linear Regression

The simplest regression analysis algorithm. Uses backward learning to train itself. 

'''python
class LinearModel():
    def __init__(self, X, y, n_iter=1000, learning_rate=0.01):
        self.w = np.zeros((X.shape[1], 1))
        self.learning_rate = learning_rate/X.shape[0]
        self.n_iter = n_iter
        self.X = X
        self.y = y
        
    def fit(self):
        for i in range(self.n_iter):
            predictions = self.X @ self.w - self.y
            delta = self.learning_rate*(self.X.T @ (predictions))
            self.w -= delta
    
    def predict(self, X):
        return X @ self.w
'''

### Logistic Regression

This algorithm is very similar to linear regression but instead of using the linear equation as the base model, it uses the sigmoid function. This can only be used for binary classification. 


### Decision Tree Regressor

This is a tree based structure that uses multiple cuts as the ideal way to getting spatial classification using the average as the output.

'''python
class DecisionTree():
    def __init__(self, x, y, tree_ids, min_leaf_samples=3):
        self.x, self.y, self.tree_ids, self.min_leaf_samples = x, y, tree_ids, min_leaf_samples
        self.n, self.ncols = len(self.tree_ids), self.x.shape[1]
        self.mse = float('inf')
        self.score = np.mean(y[self.tree_ids])
        self.find_splits()
    
    def find_splits(self):
        for i in range(self.ncols):
            self.find_better_split(i)
        if self.is_leaf: 
            return
        x = self.x.values[self.tree_ids, self.split_col]
        lhs_ids = np.nonzero(x <= self.split_val)[0]
        rhs_ids = np.nonzero(x > self.split_val)[0]
        self.lhs = DecisionTree(self.x, self.y, self.tree_ids[lhs_ids])
        self.rhs = DecisionTree(self.x, self.y, self.tree_ids[rhs_ids])
        
    def find_better_split(self, column):
        x = self.x.values[self.tree_ids, column]
        y = self.y[self.tree_ids]
        sorted_indxs = np.argsort(x)
        x_sort, y_sort = x[sorted_indxs], y[sorted_indxs]
        
        rhs_cnt = self.n
        lhs_cnt = 0
        
        rhs_sum = y_sort.sum()
        lhs_sum = 0
        
        rhs_sum2 = (y_sort**2).sum()
        lhs_sum2 = 0
        
        for i in range(0, self.n-self.min_leaf_samples-1):
            lhs_cnt += 1
            rhs_cnt -= 1
            lhs_sum += y_sort[i]
            rhs_sum -= y_sort[i]
            lhs_sum2 += y_sort[i]**2
            rhs_sum2 -= y_sort[i]**2
            if i < self.min_leaf_samples or x_sort[i]==x_sort[i+1]:
                continue
            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_cnt*lhs_std + rhs_cnt*rhs_std
            if curr_score < self.mse:
                self.mse, self.split_val, self.split_col = curr_score, x_sort[i], column
    @property       
    def is_leaf(self):
        return self.mse == float('inf')
    
    def predict(self,x):
        return [self.predict_row(row) for index,row in x.iterrows()]
    
    def predict_row(self, row):
        if self.is_leaf:
            return self.score
        if row[self.split_col] < self.split_val:
            return self.lhs.predict_row(row)
        else: return self.rhs.predict_row(row)
    
    def __repr__(self):
        output = f'samples: {self.n}, value: {self.score}'
        if not self.is_leaf:
            output += f' column: {self.x.columns[self.split_col]}, value: {self.split_val}'
        return output
        
'''

### Random Forest Regressor

This is an ensemble based classifier that builds on top of the decision tree class.

'''python
class RandomForestR():
    def __init__(self, x, y, n_trees=10, min_leaf_samples=3, n_samples=50):
        np.random.seed(42)
        self.x, self.y, self.min_leaf_samples, self.n_samples = x, y, min_leaf_samples, n_samples
        self.trees = []
        for i in range(n_trees):
            rnd_ids = np.random.permutation(len(self.y))[:self.n_samples]
            self.trees.append(DecisionTree(self.x.iloc[rnd_ids], self.y[rnd_ids], np.array(range(len(rnd_ids))), self.min_leaf_samples))
    
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)
    
    def score(self, x, y):
        return r2_score(y, self.predict(x))

'''

## License

MIT License