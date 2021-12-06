from collections import defaultdict
from helpers import Dataset
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
from collections import namedtuple

FakeState = namedtuple("FakeState", "name")

class MFCTagger:

    missing = FakeState(name="<MISSING>")

    def __init__(self, table):
        self.table = defaultdict(lambda: MFCTagger.missing)
        self.table.update({word: FakeState(name=tag) for word, tag in table.items()})

    def viterbi(self, seq):
        """This method simplifies predictions by matching the Pomegranate viterbi() interface"""
        return 0., list(enumerate(["<start>"] + [self.table[w] for w in seq] + ["<end>"]))

def pair_counts(sequences_A, sequences_B):
    """Return a dictionary keyed to each unique value in the first sequence list
    that counts the number of occurrences of the corresponding value from the
    second sequences list.

    For example, if sequences_A is tags and sequences_B is the corresponding
    words, then if 1244 sequences contain the word "time" tagged as a NOUN, then
    you should return a dictionary such that pair_counts[NOUN][time] == 1244
    """

    pair_counts_dict = defaultdict(lambda: defaultdict(int))
    zipped_seqs = zip(sequences_A, sequences_B)

    for A, B in zipped_seqs:

        pair_counts_dict[A][B] += 1

    return pair_counts_dict

def replace_unknown(sequence):
    """Return a copy of the input sequence where each unknown word is replaced
    by the literal string value 'nan'. Pomegranate will ignore these values
    during computation.
    """
    return [w if w in data.training_set.vocab else 'nan' for w in sequence]

def simplify_decoding(X, model):
    """X should be a 1-D sequence of observations for the model to predict"""

    _, state_path = model.viterbi(replace_unknown(X))

    return [state[1].name for state in state_path[1:-1]]


def accuracy(X, Y, model):
    """Calculate the prediction accuracy by using the model to decode each sequence
    in the input X and comparing the prediction with the true labels in Y.

    The X should be an array whose first dimension is the number of sentences to test,
    and each element of the array should be an iterable of the words in the sequence.
    The arrays X and Y should have the exact same shape.

    X = [("See", "Spot", "run"), ("Run", "Spot", "run", "fast"), ...]
    Y = [(), (), ...]
    """
    correct = total_predictions = 0

    for observations, actual_tags in zip(X, Y):

        # The model.viterbi call in simplify_decoding will return None if the HMM
        # raises an error (for example, if a test sentence contains a word that
        # is out of vocabulary for the training set). Any exception counts the
        # full sentence as an error (which makes this a conservative estimate).

        try:
            most_likely_tags = simplify_decoding(observations, model)
            correct += sum(p == t for p, t in zip(most_likely_tags, actual_tags))

        except:

            pass

        total_predictions += len(observations)

    return correct / total_predictions


def unigram_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequence list that
    counts the number of occurrences of the value in the sequences list. The sequences
    collection should be a 2-dimensional array.

    For example, if the tag NOUN appears 275558 times over all the input sequences,
    then you should return a dictionary such that your_unigram_counts[NOUN] == 275558.
    """

    output_dict = defaultdict(int)

    for s in sequences:

        output_dict[s] += 1

    return output_dict


def bigram_counts(sequences):
    """Return a dictionary keyed to each unique PAIR of values in the input sequences
    list that counts the number of occurrences of pair in the sequences list. The input
    should be a 2-dimensional array.

    For example, if the pair of tags (NOUN, VERB) appear 61582 times, then you should
    return a dictionary such that your_bigram_counts[(NOUN, VERB)] == 61582
    """

    output_dict = defaultdict(int)

    for osdx in range(len(sequences)):

        for isdx in range(len(sequences[osdx]) - 1):

            bigram = (sequences[osdx][isdx], sequences[osdx][isdx + 1])
            output_dict[bigram] += 1

    return output_dict


def starting_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the beginning of
    a sequence.

    For example, if 8093 sequences start with NOUN, then you should return a
    dictionary such that your_starting_counts[NOUN] == 8093
    """

    first_word_only = [s[0] for s in sequences]
    starting_counts = defaultdict(int)

    for fw in first_word_only:

        starting_counts[fw] += 1

    return starting_counts


def ending_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the end of
    a sequence.

    For example, if 18 sequences end with DET, then you should return a
    dictionary such that your_starting_counts[DET] == 18
    """

    last_word_only = [s[-1] for s in sequences]
    end_counts = defaultdict(int)

    for lw in last_word_only:

        end_counts[lw] += 1

    return end_counts

if __name__=='__main__':

    data = Dataset("tags-universal.txt", "brown-universal.txt", train_test_split=0.8)

    print("There are {} sentences in the corpus.".format(len(data)))
    print("There are {} sentences in the training set.".format(len(data.training_set)))
    print("There are {} sentences in the testing set.".format(len(data.testing_set)))
    print("There are a total of {} samples of {} unique words in the corpus."
          .format(data.N, len(data.vocab)))
    print("There are {} samples of {} unique words in the training set."
          .format(data.training_set.N, len(data.training_set.vocab)))
    print("There are {} samples of {} unique words in the testing set."
          .format(data.testing_set.N, len(data.testing_set.vocab)))
    print("There are {} words in the test set that are missing in the training set."
          .format(len(data.testing_set.vocab - data.training_set.vocab)))

    PoS_tags = [s for subseq in data.Y for s in subseq]
    words = [s for subseq in data.X for s in subseq]
    emission_counts = pair_counts(PoS_tags, words)
    PoS_tags = [t for i, (w, t) in enumerate(data.training_set.stream())]
    words = [w for i, (w, t) in enumerate(data.training_set.stream())]
    word_counts = pair_counts(words, PoS_tags)
    mfc_table = {}

    for w, t in word_counts.items():

        try:
            mfc_table[w]

        except:

            mfc_table[w] = 'VERB'

        for tag in list(t):

            if word_counts[w][tag] > word_counts[w][mfc_table[w]]:

                mfc_table[w] = tag

    mfc_model = MFCTagger(mfc_table)
    mfc_training_acc = accuracy(data.training_set.X, data.training_set.Y, mfc_model)

    print("training accuracy mfc_model: {:.2f}%".format(100 * mfc_training_acc))

    mfc_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, mfc_model)

    print("testing accuracy mfc_model: {:.2f}%".format(100 * mfc_testing_acc))

    tags = [t for i, (w, t) in enumerate(data.training_set.stream())]
    tag_unigrams = unigram_counts(tags)
    tag_bigrams = bigram_counts(data.training_set.Y)
    tag_starts = starting_counts(data.training_set.Y)
    tag_ends = ending_counts(data.training_set.Y)
    basic_model = HiddenMarkovModel(name="base-hmm-tagger")
    dist_states = []
    states_dict = {}

    for tag in data.training_set.tagset:

        dist = DiscreteDistribution(
            {word: emission_counts[tag][word] / float(tag_unigrams[tag]) for word in emission_counts[tag]})
        dist = State(dist, name=tag)
        dist_states.append(dist)
        states_dict[tag] = dist

    basic_model.add_states()
    Pbi = {tag: {} for tag in data.training_set.tagset}

    for bigram, counts in tag_bigrams.items():

        Pbi[bigram[0]][bigram[1]] = counts / float(tag_unigrams[bigram[0]])

    Pst = {}
    Pte = {}

    for each_state, tag in zip(dist_states, data.training_set.tagset):

        Pst[tag] = float(tag_starts[tag]) / float(sum(tag_starts.values()))
        Pte[tag] = float(tag_ends[tag]) / float(tag_unigrams[tag])
        basic_model.add_transition(basic_model.start, each_state, Pst[tag])
        basic_model.add_transition(each_state, basic_model.end, Pte[tag])

    for last_tag, next_tag in tag_bigrams.keys():

        basic_model.add_transition(states_dict[last_tag], states_dict[next_tag], Pbi[last_tag][next_tag])

    basic_model.bake()
    hmm_training_acc = accuracy(data.training_set.X, data.training_set.Y, basic_model)

    print("training accuracy basic hmm model: {:.2f}%".format(100 * hmm_training_acc))

    hmm_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, basic_model)

    print("testing accuracy basic hmm model: {:.2f}%".format(100 * hmm_testing_acc))