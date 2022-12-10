import argparse
import numpy as np

# python learnhmm.py toy_data/train.txt toy_data/index_to_word.txt toy_data/index_to_tag.txt toy_data/hmminit.txt toy_data/hmmemit.txt toy_data/hmmtrans.txt
# python learnhmm.py en_data/train.txt en_data/index_to_word.txt en_data/index_to_tag.txt en_data/hmminit.txt en_data/hmmemit.txt en_data/hmmtrans.txt
# python learnhmm.py fr_data/train.txt fr_data/index_to_word.txt fr_data/index_to_tag.txt fr_data/hmminit.txt fr_data/hmmemit.txt fr_data/hmmtrans.txt


def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    
    Where above the arguments have the following types:

        train_data --> A list of training examples, where each training example is a list
            of tuples train_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        init_out --> A file path to which you should write your initial probabilities

        emit_out --> A file path to which you should write your emission probabilities

        trans_out --> A file path to which you should write your transition probabilities
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmmprior", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)

    args = parser.parse_args()

    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    return train_data, words_to_indices, tags_to_indices, args.hmmprior, args.hmmemit, args.hmmtrans


if __name__ == "__main__":
    # Collect the input data
    train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()     
    # Initialize the initial, emission, and transition matrices
    initial = np.zeros((len(tags_to_index), 1)) # 1 by num_tags
    transition = np.zeros((len(tags_to_index), len(tags_to_index))) # num_tags * num_tags
    emission = np.zeros((len(tags_to_index), len(words_to_index))) # num_tags * num_words 
    # Increment the matrices
    for i in range(0, len(train_data)):
        for j in range(0, len(train_data[i])):
            word, tag = train_data[i][j]
            word_ind = words_to_index[word]
            tag_ind = tags_to_index[tag]
            if (j == 0):
                initial[tag_ind] += 1
            if (j != len(train_data[i])-1):
                word2, tag2 = train_data[i][j+1]
                tag2_ind = tags_to_index[tag2]
                transition[tag_ind][tag2_ind] += 1
            emission[tag_ind][word_ind] += 1
    # Add a pseudocount
    initial = initial + 1
    transition = transition + 1
    emission = emission + 1

    # Normalize matrices 
    init_sums = np.sum(initial)
    trans_sums = np.sum(transition, axis = 1) # col sums of transition
    emit_sums = np.sum(emission, axis = 1) # col sums of emission 

    initial = initial / init_sums
    
    for j in range(0, len(transition)):
        for k in range(0, len(transition[j])):
            transition[j][k] = transition[j][k] / trans_sums[j]
    #transition = transition / trans_sums

    for l in range(0, len(emission)):
        for m in range(0, len(emission[l])):
            emission[l][m] = emission[l][m] / emit_sums[l]


    # Save your matrices to the output files --- the reference solution uses 
    # np.savetxt (specify delimiter="\t" for the matrices)
    np.savetxt(init_out, initial, delimiter=" ")
    np.savetxt(emit_out, emission, delimiter=" ")
    np.savetxt(trans_out, transition, delimiter=" ")
    # Save your matrices to the output files --- the reference solution uses 
    # np.savetxt (specify delimiter="\t" for the matrices)
    pass