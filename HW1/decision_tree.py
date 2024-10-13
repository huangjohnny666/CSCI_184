# decision_tree.py
# ---------

# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree




def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector x.
    """

    # INSERT YOUR CODE HERE

    partition_dict = {}
    unique_values = np.unique(x)
    for value in unique_values:
        indices = np.where(x == value)[0]
        partition_dict[value] = indices
    return partition_dict
    # raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in y

    Returns the entropy of y: H(y) = p(y=v1)*log2(p(y=v1)) + ... + p(y=vk)*log2(p(y=vk))
    """

    # INSERT YOUR CODE HERE
    values, counts = np.unique(y, return_counts=True)
    size = counts.sum()
    entropy = 0
    for count in counts:
        p = count / size
        entropy -= p * np.log2(p)
    return entropy
    
    # raise Exception('Function not yet implemented!')


def information_gain(x, y):
    """
    Compute the information gain of a data column (x) with respect to the labels (y). The data column is a single  
    attribute/feature over all the examples (n x 1). Information gain is the difference between the entropy BEFORE 
    the split set (parent node), and the weighted-average entropy of EACH possible child node after splitting.

    Returns the information gain: IG(x, y) = parent entroy H(Parent)  - weighted sum of child entroy H(child)
    """

    # INSERT YOUR CODE HERE
    parent_entropy = entropy(y)
    partitions = partition(x)
    size = len(y)
    weighted_entropy = 0
    for idx in partitions.values():
        subset_entropy = entropy(y[idx])
        weight = len(idx) / size
        weighted_entropy += weight * subset_entropy
    return parent_entropy - weighted_entropy

    # raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    if attribute_value_pairs is None:
        attribute_value_pairs = [(i, v) for i in range(x.shape[1]) for v in np.unique(x[:, i])]
    
    # Base cases
    if np.all(y == y[0]):
        return y[0]
    if not attribute_value_pairs or depth == max_depth:
        return np.bincount(y).argmax()

    # Find the best attribute-value pair
    best_gain = -1
    best_pair = None
    for attribute, value in attribute_value_pairs:
        subset_indices = partition(x[:, attribute] == value)
        gain = information_gain(x[:, attribute] == value, y)
        if gain > best_gain:
            best_gain = gain
            best_pair = (attribute, value)
            best_indices = subset_indices

    if best_pair is None:
        return np.bincount(y).argmax()

    # Remove the chosen attribute-value pair and split
    attribute_value_pairs = [av for av in attribute_value_pairs if av != best_pair]
    true_branch = id3(x[best_indices[True]], y[best_indices[True]], attribute_value_pairs, depth + 1, max_depth)
    false_branch = id3(x[best_indices[False]], y[best_indices[False]], attribute_value_pairs, depth + 1, max_depth)

    return {(best_pair[0], best_pair[1], True): true_branch, (best_pair[0], best_pair[1], False): false_branch}


    # raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    if not isinstance(tree, dict):
        return tree
    for (attribute, value, is_true), subtree in tree.items():
        if (x[attribute] == value) == is_true:
            return predict_example(x, subtree)
    return 0
    
    # raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    return np.mean(y_true != y_pred)
    
    # raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    Modify Path to your GraphViz executable if needed. DO NOT MODIFY OTHER PART OF THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    # os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    os.environ["PATH"] += os.pathsep + '/usr/local/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid

'''
if __name__ == '__main__':
    
    #You may modify the following parts as needed
    
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    # indicating max_depth
    decision_tree = id3(Xtrn, ytrn, max_depth=5)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
'''


# A
def evaluate_depths(X_train, y_train, X_test, y_test, max_depth):
    training_errors = []
    testing_errors = []

    for depth in range(1, max_depth + 1):
        tree = id3(X_train, y_train, max_depth=depth)
        y_train_pred = [predict_example(x, tree) for x in X_train]
        y_test_pred = [predict_example(x, tree) for x in X_test]
        train_err = compute_error(y_train, y_train_pred)
        test_err = compute_error(y_test, y_test_pred)
        training_errors.append(train_err)
        testing_errors.append(test_err)
    
    return training_errors, testing_errors

def plot_errors(training_errors, testing_errors, title, filename):
    depths = range(1, len(training_errors) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(depths, training_errors, label='Training Error')
    plt.plot(depths, testing_errors, label='Testing Error')
    plt.xlabel('Depth of tree')
    plt.ylabel('Error')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # Save the plot to a file
    plt.close()  # Close the plot to avoid display issues in some environments

'''
if __name__ == '__main__':
    # Loop through each of the MONK's problems datasets
    for i in range(1, 4):
        # Construct file paths
        train_file = f'monks-{i}.train'
        test_file = f'monks-{i}.test'

        # Load the training and test datasets
        M_train = np.genfromtxt(train_file, missing_values=0, skip_header=0, delimiter=',', dtype=int)
        M_test = np.genfromtxt(test_file, missing_values=0, skip_header=0, delimiter=',', dtype=int)

        # Separate features and labels
        X_train, y_train = M_train[:, 1:], M_train[:, 0]
        X_test, y_test = M_test[:, 1:], M_test[:, 0]

        # Evaluate depths from 1 to 10
        training_errors, testing_errors = evaluate_depths(X_train, y_train, X_test, y_test, max_depth=10)

        # Plot learning curves and save the plots as images
        plot_filename = f"learning_curves_monks_problem_{i}.png"
        plot_errors(training_errors, testing_errors, f"Learning Curves for MONK's Problem {i}", plot_filename)
'''


'''
if __name__ == '__main__':
    # Loop through each of the MONK's problems datasets
    for i in range(1, 4):
        print(f"\nEvaluating and saving decision trees for MONK's problem {i}:")

        # Construct file paths
        train_file = f'monks-{i}.train'
        test_file = f'monks-{i}.test'

        # Load the training and test datasets
        M_train = np.genfromtxt(train_file, missing_values=0, skip_header=0, delimiter=',', dtype=int)
        M_test = np.genfromtxt(test_file, missing_values=0, skip_header=0, delimiter=',', dtype=int)

        # Separate features and labels
        X_train, y_train = M_train[:, 1:], M_train[:, 0]
        X_test, y_test = M_test[:, 1:], M_test[:, 0]

        # Initialize lists to store errors for plotting
        training_errors = []
        testing_errors = []

        # Evaluate depths from 1 to 10
        for depth in range(1, 11):
            # Learn a decision tree of depth 'depth'
            decision_tree = id3(X_train, y_train, max_depth=depth)

            # Commenting out the pretty print of the learned decision tree
            # print(f"\nLearned decision tree for depth={depth}:")
            # pretty_print(decision_tree)

            # Commenting out saving the learned decision tree visualization as a PNG image
            # tree_filename = f"decision_tree_monks_problem_{i}_depth_{depth}.png"
            # dot_str = to_graphviz(decision_tree)
            # render_dot_file(dot_str, tree_filename)

            # Compute the training and testing errors
            y_train_pred = [predict_example(x, decision_tree) for x in X_train]
            y_test_pred = [predict_example(x, decision_tree) for x in X_test]
            train_err = compute_error(y_train, y_train_pred)
            test_err = compute_error(y_test, y_test_pred)
            training_errors.append(train_err)
            testing_errors.append(test_err)

            # Print the errors for current depth
            print(f"Depth {depth}: Training error = {train_err:.4f}, Testing error = {test_err:.4f}")

        # Compute and print the average errors
        avg_training_error = np.mean(training_errors)
        avg_testing_error = np.mean(testing_errors)
        print(f"Average training error for MONK's problem {i}: {avg_training_error:.4f}")
        print(f"Average testing error for MONK's problem {i}: {avg_testing_error:.4f}")

        # Plot and save learning curves as images
        plot_filename = f"learning_curves_monks_problem_{i}.png"
        plot_errors(training_errors, testing_errors, f"Learning Curves for MONK's Problem {i}", plot_filename)
'''

# end of A


# B

def print_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
'''
if __name__ == '__main__':
    depths = [1, 3, 5]

    # Construct file paths
    train_file = 'monks-1.train'
    test_file = 'monks-1.test'

    # Load the training and test datasets
    M_train = np.genfromtxt(train_file, missing_values=0, skip_header=0, delimiter=',', dtype=int)
    M_test = np.genfromtxt(test_file, missing_values=0, skip_header=0, delimiter=',', dtype=int)

    # Separate features and labels
    X_train, y_train = M_train[:, 1:], M_train[:, 0]
    X_test, y_test = M_test[:, 1:], M_test[:, 0]

    # Loop over the specified depths
    for depth in depths:
        decision_tree = id3(X_train, y_train, max_depth=depth)


        print(f"\nLearned decision tree for depth={depth}:")
        pretty_print(decision_tree)
        tree_filename = f"decision_tree_monks_problem_{i}_depth_{depth}.png"
        dot_str = to_graphviz(decision_tree)
        render_dot_file(dot_str, tree_filename)

        # Compute the predictions and print the confusion matrix
        y_pred = [predict_example(x, decision_tree) for x in X_test]
        print(f"\nConfusion Matrix for depth={depth}:")
        print_confusion_matrix(y_test, y_pred, f'Confusion Matrix for MONK\'s Problem {i}, Depth {depth}')
'''
# end of B

# C



if __name__ == '__main__':
    depths = [1, 3, 5]

    # Construct file paths
    train_file = 'monks-1.train'
    test_file = 'monks-1.test'

    # Load the training and test datasets
    M_train = np.genfromtxt(train_file, missing_values=0, skip_header=0, delimiter=',', dtype=int)
    M_test = np.genfromtxt(test_file, missing_values=0, skip_header=0, delimiter=',', dtype=int)

    # Separate features and labels
    X_train, y_train = M_train[:, 1:], M_train[:, 0]
    X_test, y_test = M_test[:, 1:], M_test[:, 0]

    for depth in depths:
        # Initialize and train the DecisionTreeClassifier
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        
        
        # Print the structure of the learned decision tree
        tree_rules = export_text(clf, feature_names=['x%d' % i for i in range(len(X_train[0]))])
        print(f"Learned decision tree structure for depth={depth}:\n{tree_rules}")
        
        plt.figure(figsize=(12, 8))
        tree.plot_tree(clf, filled=True, feature_names=['x%d' % i for i in range(X_train.shape[1])],
                       class_names=["Class 0", "Class 1"])
        plt.title(f'Decision Tree for MONK\'s Problem 1, Depth {depth}')
        decision_tree_filename = f"decision_tree_monks_problem_1_depth_{depth}.png"
        plt.savefig(decision_tree_filename)
        plt.close()

        # Predict the labels for the test set
        y_pred = clf.predict(X_test)
        
        # Compute the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix for depth={depth}:\n{cm}")
        
        # Visualize and save the confusion matrix as an image
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for MONK\'s Problem 1, Depth {depth}')
        confusion_matrix_filename = f"confusion_matrix_monks_problem_1_depth_{depth}.png"
        plt.savefig(confusion_matrix_filename)
        plt.close()
        
        # Visualize and save the decision tree as an image
        plt.figure(figsize=(12, 8))
        tree.plot_tree(clf, filled=True)
        plt.title(f'Decision Tree for MONK\'s Problem 1, Depth {depth}')
        decision_tree_filename = f"decision_tree_monks_problem_1_depth_{depth}.png"
        plt.savefig(decision_tree_filename)
        plt.close()

