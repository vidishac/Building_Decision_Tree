class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self, attr, v):
        self.attribute = attr
        self.left = None
        self.right = None
        self.vote = v
        self.depth = 0
        self.leftedge = None
        self.rightedge = None
        self.stats = {}

'''
Helper Functions  
'''
def entropy(inp):

    labels, counts = np.unique(inp[:,-1], return_counts=True)
    if len(labels) == 1:
        return 0
    else:
        return -(counts[0]/len(inp))*m.log2((counts[0]/len(inp))) - (
            counts[1]/len(inp))*m.log2((counts[1]/len(inp)))
    
def mut_info(inp):
   
    features = inp.shape[1]-1
    array = np.zeros(features)
    for f in range(features):
        labels, counts = np.unique(inp[:,f], return_counts=True)
        if len(labels) == 2:
            inp_0 = inp[inp[:,f]== labels[0],]
            inp_1 = inp[inp[:,f]== labels[1],]
            array[f] = entropy(inp) - entropy(inp_0)*(counts[0]/len(inp)) - entropy(inp_1)*(
                        counts[1]/len(inp))
    return array

def majority_label(inp):
    

    labels, counts = np.unique(inp[:,-1], return_counts=True)
    if len(labels) == 1: 
        return labels[0]
        
    else:
        if counts[0] == counts[1]:
            return np.sort(labels)[1]
        else:
            return labels[np.argmax(counts)]



def decision_tree_learner(train_inp, max_depth, depth = 0):

    
    labels, counts = np.unique(train_inp[:,-1], return_counts=True)
    
    '''
    Base Case
    1)If there's just one type of label in the whole data
   
    '''
    if len(labels) == 1: 
        root = Node(None, majority_label(train_inp))
        root.stats = root.stats = {labels[0]:counts[0]}
        return root
    
    '''
    Base Case
    2)If maximum depth allowed is 0
    '''
    num_f = train_inp.shape[1]-1
    
    if max_depth == 0 or num_f == 0:
        root = Node(None, majority_label(train_inp))
        root.stats = {labels[0]:counts[0], labels[1]:counts[1]}
        return root
    
    else: 
        

        if max(mut_info(train_inp))>0 and depth < max_depth and depth < num_f:
            split_arg = np.argmax(mut_info(train_inp))
            root = Node(header[split_arg],None)
            root.depth = depth
            root.stats = {labels[0]:counts[0], labels[1]:counts[1]}
            
            
            '''
            Splitting data
            '''
            labels_f= np.unique(train_inp[:,split_arg])
            labels_f= np.sort(labels_f)
            train_inp_0 = train_inp[train_inp[:,split_arg]== labels_f[0],]
            train_inp_1 = train_inp[train_inp[:,split_arg]== labels_f[1],]
            
            
            root.leftedge = labels_f[0]
            root.left = decision_tree_learner(train_inp_0, max_depth, depth + 1)
            root.left.depth = depth + 1
            root.rightedge = labels_f[1]
            root.right = decision_tree_learner(train_inp_1, max_depth, depth + 1)
            root.right.depth = depth + 1

            return root
            
        else: # Base case when all atributes have the same labels(max(mut_info(train_inp))=0)     
        
            root = Node(None, majority_label(train_inp))
            root.stats = {labels[0]:counts[0], labels[1]:counts[1]}
            return root

def printing_tree(root, first = True):
    
    if first:
        print('[{} {} / {} {}]'.format(list(root.stats.values())[0],list(root.stats.keys())[0], 
                                   list(root.stats.values())[1], list(root.stats.keys())[1]))
        
        
    if root.attribute is not None :
        
        if len(root.left.stats) == 2:
    
            print('{} {} = {} : [{} {} / {} {}] '.format('|'*(root.depth + 1), root.attribute, 
                                                        root.leftedge, list(root.left.stats.values())[0],
                                                        list(root.left.stats.keys())[0],
                                                        list(root.left.stats.values())[1],
                                                        list(root.left.stats.keys())[1]))  
        else:
            print('{} {} = {} : [{} {}] '.format('|'*(root.depth + 1), root.attribute, 
                                                        root.leftedge, 
                                                        list(root.left.stats.values())[0],
                                                        list(root.left.stats.keys())[0]))  
            
        if root.left is not None:
            printing_tree(root.left, False)

        if len(root.right.stats) == 2:
    
            print('{} {} = {} : [{} {} / {} {}] '.format('|'*(root.depth + 1), root.attribute, 
                                                        root.rightedge, list(root.right.stats.values())[0],
                                                        list(root.right.stats.keys())[0],
                                                        list(root.right.stats.values())[1],
                                                        list(root.right.stats.keys())[1]))  
        else:
            print('{} {} = {} : [{} {}] '.format('|'*(root.depth + 1), root.attribute, 
                                                        root.rightedge, 
                                                        list(root.right.stats.values())[0],
                                                        list(root.right.stats.keys())[0]))  
        if root.right is not None:
            printing_tree(root.right, False)



def predict(root, example):
    
    if root.vote is not None:
        return root.vote
    
    elif example[header == root.attribute] == root.leftedge:
        return predict(root.left, example)
        
    elif example[header == root.attribute] == root.rightedge:
        return predict(root.right, example)
            
    else:
        return majority_label(train_input)
    

def decision_tree_predict(decision_tree,inp,out):

    with open(out, 'w', newline= '\n') as output:      
        for line in inp[:,:-1]:
            output.write(str(predict(decision_tree, line)) + '\n')


def error_rate(decision_tree,inp):
    
    error_count = 0
    
    for line in inp:
        if line[-1] != predict(decision_tree, line[:-1]):
            error_count += 1
    return error_count/len(inp)
   

def decision_tree_metrics(decision_tree,train_inp,test_inp,metrics_out):

    errors = ['error(train): ',str(error_rate(decision_tree,train_inp)),'\n', 'error(test): ', 
              str(error_rate(decision_tree,test_inp)),'\n']

    with open(metrics_out, 'w', newline= '\n') as output:    
            output.writelines(errors)

    
    
        

if __name__ == '__main__':
    
    import sys
    import numpy as np
    import math as m
    
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_output = sys.argv[4]
    test_output = sys.argv[5]
    metrics_output = sys.argv[6]
    
    
    
    train_input = np.genfromtxt(train_input, delimiter='\t', dtype = 'str')
    header = train_input[0,:-1]
    train_input = train_input[1:,]
    
    test_input = np.genfromtxt(test_input, delimiter='\t', dtype = 'str', skip_header=1)
    
    assert train_input.shape[1] == test_input.shape[1]
    
    '''
    Build the tree
    '''
    
    decision_tree = decision_tree_learner(train_input, max_depth)
    
    '''
    Make predictions on training and test data and save to output files
    '''
    
    decision_tree_predict(decision_tree, train_input, train_output)
    decision_tree_predict(decision_tree, test_input, test_output)
    
    '''
    Calculate train and test errors and save to output files 
    '''
    decision_tree_metrics(decision_tree,train_input,test_input,metrics_output)
    
    
    '''
    Printing the tree
    '''     
    printing_tree(decision_tree)
    
  
    
  
    '''
    Testing
    '''    
    
    #Using Command Prompt

    #  cd C:\Users\vidis\Documents\Carnegie Mellon University\Machine Learning\hw2\handout
    
    #  python decision_tree.py education_train.tsv education_test.tsv 3 edu_3_train.labels edu_3_test.labels edu_3_metrics.txt
    
    
    
    
    '''
  
    
    train_input = 'education_train.tsv'
    test_input = 'education_test.tsv'


    def error_array(inp): 
        f_num = inp.shape[1]
        print(f_num)
        array = np.zeros(f_num)
        depths = np.arange(f_num)
        for i in range(f_num):
            array[i] = error_rate(decision_tree_learner(train_input, i), inp)
            print(array[i])
        return (array,depths)

    train_array = error_array(train_input)[0]
    test_array = error_array(test_input)[0]
    depth_array = error_array(train_input)[1]

    import matplotlib.pyplot as plt
    


    plt.plot(depth_array,train_array, label= 'Train Error')
    plt.plot(depth_array,test_array, label= 'Test Error')
    plt.legend(loc="upper right")
    plt.savefig('education.png', dpi = 100)
    plt.show()
    
    '''






