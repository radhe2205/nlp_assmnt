import math

emission_probs = {
    "NN": {
        "time": 0.070727, "arrow": 0.000215
    },
    "VB": {
        "time": 0.000005, "like": 0.028413
    },
    "JJ": {
        "time": 0.
    },
    "VBZ": {
        "flies": 0.004754
    },
    "NNS": {
        "flies": 0.001610
    },
    "IN": {
        "like": 0.026512
    },
    "RB": {
        "like": 0.005086
    },
    "DT": {
        "an": 0.014192
    }
}
transition_probs = {
    "S": {
        "NN": 0.006823, "VB": 0.005294, "JJ": 0.008033
    },
    "NN": {
        "VBZ": 0.039005, "NNS": 0.016076, "E": 0.002069
    },
    "VB":{
        "VBZ": 0.000566, "NNS": 0.006566, "DT": 0.152649
    },
    "JJ": {
        "VBZ": 0.020934, "NNS": 0.024383
    },
    "VBZ": {
        "IN": 0.085862, "VB": 0.007002, "RB": 0.150350
    },
    "NNS": {
        "IN": 0.218302, "VB": 0.111406, "RB": 0.064721
    },
    "IN": {
        "DT": 0.314263
    },
    "RB": {
        "DT": 0.053113
    },
    "DT": {
        "NN": 0.380170
    }
}

def calculate_sequence_prob(tag_seq, word_seq):
    tag_seq = ["S"] + tag_seq + ["E"]
    prob = 0
    for i in range(len(word_seq)):
        if transition_probs[tag_seq[i]][tag_seq[i+1]] == 0 or emission_probs[tag_seq[i+1]][word_seq[i]] == 0:
            return math.inf
        prob -= math.log(transition_probs[tag_seq[i]][tag_seq[i+1]])
        prob -= math.log(emission_probs[tag_seq[i+1]][word_seq[i]])
    return prob - math.log(transition_probs[tag_seq[-2]][tag_seq[-1]])

print(calculate_sequence_prob(["VB", "NNS", "IN", "DT", "NN"], "time flies like an arrow".split()))
print(calculate_sequence_prob(["JJ", "VBZ", "VB", "DT", "NN"], "time flies like an arrow".split()))
