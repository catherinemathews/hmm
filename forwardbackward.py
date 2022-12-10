import argparse
import numpy as np

def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("validation_data", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)

    args = parser.parse_args()

    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix

def logsumexp(v):
    m = np.max(v, axis=1)
    res = np.zeros(len(v))
    for i in range(0, len(v)):
        
        res[i] = m[i] + np.log(np.sum(np.exp(v[i] - m[i])))
    return res

def logsumexp_vec(v):
    m = max(v)
    e = np.exp(v-m)
    return m + np.log(np.sum(e))


def forwardbackward(seq, loginit, logtrans, logemit, words_to_indices):
    """
    Your implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix
    
    You should compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    T = len(seq) # number of words (T?)
    J = len(loginit) # number of hidden states (J?)
    
    # Initialize log_alpha and fill it in
    log_alpha = np.zeros((T, J))
    for t in range(0, T):
        ind = words_to_indices[seq[t]]
        if (t == 0):
            log_alpha[t] = loginit + logemit[:, ind]
        else:
            log_alpha[t] = logemit[:, ind] + logsumexp(log_alpha[t-1, :] + np.transpose(logtrans))
    # Initialize log_beta and fill it in
    log_beta = np.zeros((T, J))
    for j in range(0, J):
        log_beta[0][j] = 0

    for t in range(T-2, -1, -1):
        ind = words_to_indices[seq[t+1]]
        log_beta[t] = logsumexp(logemit[:, ind] + log_beta[t+1, :] + logtrans)
            #print(logemit[:, ind].shape, log_beta[t+1, :].shape, logtrans.shape)
    print(log_alpha)
    print(log_beta)
    # Compute the predicted tags for the sequence 
    probs = log_alpha + log_beta
    tags = np.argmax(probs, axis=1)
    # Compute the log-probability of the sequence - how 
    log_prob = logsumexp_vec(log_alpha[T-1, :])
    # Return the predicted tags and the log-probability
    return tags, log_prob
    

    
if __name__ == "__main__": 
    # Get the input data
    val_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, pred_file, metrics_file = get_inputs()
    # For each sequence, run forward_backward to get the predicted tags and 
    # the log-probability of that sequence.
    orig_tags = []
    pred_tags = []
    act_tags = []
    words = []
    log_likelihoods = []
    for i in range(0, len(val_data)): 
        seq = []
        for j in range(0, len(val_data[i])):
            word = val_data[i][j][0]
            seq.append(word)
            words.append(word)
            orig_tags.append(val_data[i][j][1])
        orig_tags.append("")
        words.append(" ")
        tags, probs = forwardbackward(seq, np.log(hmminit), np.log(hmmtrans), np.log(hmmemit), words_to_indices)
        pred_tags.append(tags)
        pred_tags.append([])
        log_likelihoods.append(probs)
    for i in range(0, len(pred_tags)): 
        curr_tags = pred_tags[i]
        #print(curr_tags)
        if len(curr_tags) == 0:
            act_tags.append("")
        else:
            for j in curr_tags:
                act_tag = list(tags_to_indices.keys())[j]
                act_tags.append(act_tag)
    # Compute the average log-likelihood and the accuracy. The average log-likelihood 
    # is just the average of the log-likelihood over all sequences. 
    avg_likelihood = np.mean(log_likelihoods)
    # The accuracy is the total number of correct tags across all sequences divided by the total number 
    # of tags across all sequences. 
    incorrect = 0
    correct = 0
    origs = []
    acts = []
    for i in range(0, len(orig_tags)):
        if orig_tags[i] != act_tags[i] and act_tags[i] != "":
            incorrect += 1
        else:
            if act_tags[i] != "":
                correct += 1
    accuracy = correct / (correct + incorrect) 
    with open(pred_file, 'w') as pred_out:
        for i in range(0, len(words)):
            word = words[i]
            if (word == " "):
                pred_out.write("\n")
            else:
                tag = act_tags[i]
                pred_out.write(word)
                pred_out.write("\t")
                pred_out.write(tag)
                pred_out.write("\n")
    with open(metrics_file, 'w') as met_out:
        met_out.write("Average Log-Likelihood: ")
        print(avg_likelihood)
        met_out.write(str(avg_likelihood))
        met_out.write("\n")
        met_out.write("Accuracy: ")
        met_out.write(str(accuracy))


# python forwardbackward.py toy_data/validation.txt toy_data/index_to_word.txt toy_data/index_to_tag.txt toy_data/hmminit.txt toy_data/hmmemit.txt toy_data/hmmtrans.txt toy_data/predicted.txt toy_data/metrics.txt
# python forwardbackward.py en_data/validation.txt en_data/index_to_word.txt en_data/index_to_tag.txt en_data/hmminit.txt en_data/hmmemit.txt en_data/hmmtrans.txt en_data/predicted.txt en_data/metrics.txt
# python forwardbackward.py fr_data/validation.txt fr_data/index_to_word.txt fr_data/index_to_tag.txt fr_data/hmminit.txt fr_data/hmmemit.txt fr_data/hmmtrans.txt fr_data/predicted.txt fr_data/metrics.txt