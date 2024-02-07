import numpy as np
from collections import Counter

class decision_Tree:

    def __init__(self):
        self.no_of_unique = 11
        self.counter = 0
        self.min_samples = 100
        self.max_depth = 100

    def __str__(self):
        return "DecisionTree"

    def type_of_cols(self,data):
        # initialize lists to store column type
        col_type=[]
        # move in each column
        for col in data.columns[:-1]:
            # get number of unique values in the feature
            no_of_col_unique= data[col].nunique()

            # check for categorical feature
            if data[col].dtypes==object or no_of_col_unique <= self.no_of_unique:
                col_type.append("categorical")

            # if not categorical then it is a continuous feature
            else:
                col_type.append("continuous")
        return col_type

    # the function is used to get all possible splits for the data
    def get_splits(self,data):

        splits = {} # dictonary That saves the possible splits of our data based on data-type
        feature_type=self.type_of_cols(data) # Get data-type categorical or continuous for each column

        # moving in for each feature excluding the last column which is the label
        for column_index in range(data.shape[1]-1):
            values = data.iloc[:, column_index] # getting all the feature value based on the feature names
            unique_values = np.unique(values) #fetch the unique values of the respective features
            type_of_feature = feature_type[column_index] # get data-type of column i.e categorial or continues

            # handling spkit for continuous data
            if type_of_feature == "continuous":
                splits[column_index] = []

                # passing through all unique values
                for index in range(len(unique_values)):
                  if index != 0:
                        current_value = unique_values[index]
                        previous_value = unique_values[index - 1]
                        split = (current_value + previous_value) / 2 # performing split
                        splits[column_index].append(split) #saving the split
            # for a categorical feature ther must be at least 2 unique values, else in the
            # split_data function data_below would contain all data points and data_above would be None
            # thus checking for the above mentioned situation as well
            elif len(unique_values) > 1:
                splits[column_index] = unique_values
        # return splits
        return splits

    def split_data(self,data, column, value):

        #based on to column's data type we will destribute the data into two partitions
        feature_type=self.type_of_cols(data)
        split_values = data.iloc[:,column]
        type_of_feature = feature_type[column]

        # in case of continues data we will use greater or lesser than operator
        if type_of_feature == "continuous":
            left = data[split_values <= value]
            right = data[split_values >  value]

        # in case of categorial data will use logical equal or not equal operator
        else:
            left = data[split_values == value]
            right = data[split_values != value]

        return left,right

    #functions to calculate the entropy
    def entropy(self,data):
        # we are calculating the entropy
        prob=list(dict(data.iloc[:, -1].value_counts(normalize=True)).values())
        entropy = sum(prob* -np.log2(prob))
        return entropy

    # entropy of sub-splits tree
    def entropy_data(self,left,right):
        n = len(left) + len(right)
        p_left = len(left) / n
        p_right = len(right) / n
        entropy_ =  (p_left * self.entropy(left)+ p_right *self.entropy(right))
        return entropy_

    # based on the entropy, calculate the best splits
    def best_split(self,data,splits):

        entropy = 99999
        # iterating over the splits obtained by the get_split method for each feature
        for col in splits:
            # iterating over the splits of a indivisual feature
            for val in splits[col]:

                # spliting the data according to obtained split(val) of a column
                left, right = self.split_data(data, column=col, value=val)

                # calculating the entropy for the column
                current_entropy = self.entropy_data(left,right)
                # in case where obtained entropy is lesser than assumed entropy
                # then assume the obtained entropy as best entropy
                # and the current column can be termend as best column and the split too
                if current_entropy <= entropy:
                    entropy = current_entropy
                    best_column = col
                    best_split = val

        # return best split and column
        return best_column, best_split

    # for the next step let's work on a function to build our decision tree
    # it is the base function for fitting the data
    def _tree_builder(self,df):

        # data preparations
        column=df.columns #store the column name
        feature_type=self.type_of_cols(df) # store the column value type i.e categorial/continues
        data = df

        # base case for recursion
        # checking for hyper-parameter conditions
        if  (df.iloc[:,-1].nunique()==1) or (len(data) < self.min_samples) or (self.counter == self.max_depth):
            classes= Counter(df.iloc[:,-1]).most_common(1)[0][0]
            return classes

        # recursive part
        else:
            self.counter += 1
            splits = self.get_splits(data)# calculating the splits of each columns
            split_column, split_value = self.best_split(data,splits) # getting the best column and split value
            left, right = self.split_data(data, split_column, split_value) # based on the above split and column divide the data
            # for better visualization of tree
            # after training we can see the tree in the {object}.tree variable
            # determine question
            # here we are trying to ask the question if our feature and split value have completed the classification task or not
            # also adding to this we also keep track of the base while recursion

            feature_name = column[split_column] # pick the column name
            type_of_feature = feature_type[split_column] # get type of the column
            # in case of a feature with continuous values we have right and left sub tree based on condition
            # The values in the dataset of the particular feature are less than or equals to the best split value
            if type_of_feature == "continuous":
                question = "{} <= {}".format(feature_name, split_value)

            # feature is categorical
            # we have a question if the values from the dataset is equal to the optimal categorical split value or not
            else:
                question = "{} = {}".format(feature_name, split_value)

            mytree = {question: []}


            # work on building right and left sub trees
            ans_yes = self._tree_builder(left) # left leave is for yes where tree traversal stops
            ans_no = self._tree_builder(right) # right leave needs few more nodes


            # If the answers are the same, no need to append
            if ans_yes == ans_no:
                mytree = ans_yes
            else:
                mytree[question].append(ans_yes)
                mytree[question].append(ans_no)
            return mytree

    def cfit(self,X,y):

        # sending complete data as input with last column as our target feature
        X["output"]=y

        # calling the tree builder function
        self.tree=self._tree_builder(X)
        return self.tree

    # base predict function
    def _predict(sf,dx,tree):
        root_node = list(tree.keys())[0] # fetch the root node's value i.e our dict keys which consits of column name,operator and split value
        column,operator,split=root_node.split(" ") # we have used the space as a seprator b/w the three data
        if operator == "<=": # if the operator is lesser than or equal it means th column type is continues

            if dx[column] <= float(split):
                result = tree[root_node][0]
            else:
                result = tree[root_node][1]

        # if column  is categorical then we can use logical equal operator
        else:
            if str(dx[column]) == split:
                result = tree[root_node][0]
            else:
                result = tree[root_node][1]

        if type(result)!=dict: # if the result is dict then we have more nodes to be traversed
            return result
        # else recursively travers the entire tree for accurate results
        else:
            return sf._predict(dx,result)

    def cpredict(self,X_test):
        s=[]
        for i in range(X_test.shape[0]):
            s.append(self._predict(X_test.iloc[i],self.tree))

        return s
