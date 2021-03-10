import numpy as np
	
class BinaryDecisionTree:	
	def fit(self, data, target, max_depth=0):
		self.data = data
		self.target = target
		self.max_depth = max_depth
		used_index = []
		self.root_node = self.createNode(self.data, self.target, used_index)
	
	#We use this function to train our model. Since the tree is created recursively
    #we needed a new function instead of putting this code in "registerInitialState".
    #This function returns either a node object or None.
        def createNode(self, data_node, target_node, index_used):
                #Get the size of the data input
                N = len(data_node)
                #If the length is 1 that means there is only one instance for that feature
                #so we have reached a leaf and we return no node object.
                if N <= 1:
                    return None

                #Check if the length of the tree is not bigger than the number of features.
                #If it is, return none.
                if len(index_used) >24:
                    return None
                
                #Check if the length of the tree is not bigger than the desired depth of the tree.
                #If it is, return none.
                if len(index_used) >=self.max_depth and self.max_depth>0:
                    return None
                    
                #Create 2 arrays, one for the 0's for each feature in all the instances, and one for the 1's
                count0 = [[0.0 for x in range(len(set(self.target)))] for y in range(len(data_node[0]))]
                count1 = [[0.0 for x in range(len(set(self.target)))] for y in range(len(data_node[0]))]

                #Populate the 2d arrays created before
                for i in range(len(data_node[0])):
                    for j in range(N):
                        if(data_node[j][i]==0):
                            count0[i][target_node[j]] += 1
                        if(data_node[j][i]==1):
                            count1[i][target_node[j]] += 1

                #Calculate the gini impurity for each class
                giniImpurity = 1.1
                index = -1
                for i in range(len(count1)):
                    #Check if the feature has not already been evaluated
                    if (i not in index_used):
                        newGiniImpurity = 0.0
                        sum0 = sum(count0[i])
                        sum1 = sum(count1[i])

                        newGiniImpurity0 = 1.0
                        newGiniImpurity1 = 1.0
                        for j in range(len(set(target_node))):
                            if(sum0!=0):
                                newGiniImpurity0 = newGiniImpurity0 - ((count0[i][j]/sum0)**2)
                            if(sum1!=0):
                                newGiniImpurity1 = newGiniImpurity1 - ((count1[i][j]/sum1)**2)
                        newGiniImpurity = (sum0/N)*newGiniImpurity0 + (sum1/N)*newGiniImpurity1
                        #Select the feature with the smallest gini impurity
                        if (newGiniImpurity<giniImpurity):
                            giniImpurity=newGiniImpurity
                            index = i
                #Create a new node object, with the feature with lowest gini impurity as the
                #feature_index, and the count of target_node array as the samples_per_class.
                #The right and left node are initialized as None, but are later on created by
                #this function.
                node = Node(
                    gini_impurity=giniImpurity,
                    feature_index = index,
                    samples_per_class = [target_node.count(x) for x in set(self.target)],
                    right_node = None,
                    left_node = None
                )
                left_data = []
                left_target = []
                right_data = []
                right_target = []
                #The data is split, so that there is one array containing all the instances
                #that have 1 in the selected feature and another array with all the instances
                #that have 0 in the selected feature.
                for j in range(len(target_node)):
                    if(data_node[j][index]==1):
                        left_data.append(data_node[j])
                        left_target.append(target_node[j])
                    if(data_node[j][index]==0):
                        right_data.append(data_node[j])
                        right_target.append(target_node[j])
                list_index = list(index_used)
                list_index.append(index)
                #Here, the right and left nodes are created by calling recursively this function
                #and passing the recently created arrays. The left node gets all the instances with
                #1 in the selected feature, and the right node all the instances with 0.
                node.left_node = self.createNode(left_data, left_target, list_index)
                node.right_node = self.createNode(right_data, right_target, list_index)

                return node

        #A predict function. Used after we have trained our model. This will be called each time 
        #in "getAction" to output the new move predicted by our tree for pacman.
        def predict(self, input):
                input = np.array(input)
                if input.ndim != 2:
                        raise TypeError("Only 2 dimensional arrays or lists are allowed.")
                #We asign to a variable the root node of our tree
                node = self.root_node
                pred_arr = np.array([0]*input.shape[0])
                #We use a while to run throug the tree. We get out of the while until we have reached
                #a leaf of the tree, this is, when the node doesn't have left nor right subsequent nodes
                for i, case in enumerate(input):
                        while True:
                                #First we check if the feature being compared in the node is 0. If it is, it means
                                #we will go to the right child node of the current node. We go right because
                                #we defined in our "createNode" function that 0 goes right and 1 goes left.
                                if (case[node.feature_index]==0) and (node.right_node is not None):
                                    node = node.right_node
                                else:
                                    #Then we check if the feature is 1 and there is a left node. If those
                                    #conditions are met then we will move to the left node.
                                    if (case[node.feature_index]==1) and (node.left_node is not None):
                                        node = node.left_node
                                    #If there is no right node but the feature is 0,
                                    #or if there is no left node but the feature is 1,
                                    #then we will be breaking out of the while and returning the 
                                    #predicted class for the current node.
                                    else:
                                        break
                        #The predicted class is returned. This is an integer, but later on the function
                        #"convertNumberToMove" will convert it into a move. 
                        pred_arr[i] = node.predicted_class
                return pred_arr


#We need to create a new class "Node", this will help us to create node objects in each
#call of the "createNode" function.
#The reason why this class is needed is because this way we can modify it to fit our
#requirements. For this case, the requirements are to have:
#-The calculated gini impurity in this node (this is used only for educational purposes)
#-The samples per class (to obtain the predicted class)
#-The predicted class (obtained from the max number in the samples per class)
#-The position of the feature in the instance being compared in this node
#-Left and right nodes of the same class (for the prediction and to create the tree)
class Node:
    def __init__(self, gini_impurity, samples_per_class, feature_index, right_node, left_node):
        self.gini_impurity = gini_impurity
        self.samples_per_class = samples_per_class
        self.predicted_class = samples_per_class.index(max(samples_per_class))
        self.feature_index = feature_index
        self.right_node = right_node
        self.left_node = left_node
