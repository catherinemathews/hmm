# hmm


In this project, I used Hidden Markov Models (hmms) to implement a named entity recognition system. In particular, I implemented an algorithm to learn the HMM parameters using the given training data and then implemented the forward-backward algorithm to perform a smoothing query, which was then used to predict the hidden tags for a sequence of words. 

Files:

- learnhmm.py: 
- decision_tree.py: learns a decision tree with a specified maximum depth, prints the tree's structure, predicts the labels of the training and test data, and calculates errors
- education_train.tsv / education_test.tsv: data to learn and predict grades of high school students
 -heart_train.tsv / heart_test.tsv: data to learn and predict patient's likelihood of heart disease

Command Line Arguments: python learnhmm.py < train input > < index to word > < index to tag > < hmminit > < hmmemit > < hmmtrans >

- train input: path to training input file
- index to word: path to dictionary mapping words to indices (tags are ordered by index)
- index to tag: path to dictionary mapping tags to indices 
- hmminit: path to training output file where intitalization probabilities should be written
- hmmemit: path to training output file where emission probabilities should be written
- hmmtrans: path to training output file where transition probabilities should be written
