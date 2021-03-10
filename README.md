# Binary Decision Tree
Simple implementation of a decision tree.

The reason why the name "Binary" is added is because this tree only accepts binary input features. The target or output of the tree doesn't need to be binary, it can have a wide range of values.

This decision tree uses gini impurity to decide the parent node.

This decision tree was tested in different cases. One of my favourites was a pacman game. This pacman code was developed at UC Berkeley for their AI course. The creator made the code available to everyone: http://ai.berkeley.edu/home.html

In the next image you can see the pacman code running. The decisions for the movement of pacman are decided with the binary decision tree.

![Binary tree tested in Pacman](https://github.com/CamilAbraham/Binary_Decision_Tree/blob/main/PacmanTree.PNG?raw=true)

The next image shows the accuracy achieved by the binary decision tree and the accuracy obtained with a highly developed machine learning library (scikit: sklearn tree).

![Tree accuracy](https://github.com/CamilAbraham/Binary_Decision_Tree/blob/main/TreeAccuracy.PNG?raw=true)


To do:
- [ ] Remove the binary feature. The node division will now be decided by a calculated threshold. This is so that the input features can be more than just 1 or 0.
- [ ] Add different methods for parent node selection like entropy.
