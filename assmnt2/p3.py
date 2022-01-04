import math


def process_line(line):
    puncs = ['"', "'", ",", ":", ".", ";"]
    for a in puncs:
        line = line.replace(a, " " + a + " ")

    return ["<start>"] + line.split() + ["<end>"]

def read_text(file_path):
    with open(file_path, "r") as f:
        text = f.readlines()
    return [process_line(line.lower()) for line in text]

def create_trigram_probs(lines, trigram_probs = None,
                         bigram_probs = None,
                         unigram_probs = None):

    trigram_probs = (trigram_probs, {"<meta_info>": {"count": 0, "total_tokens": 0}})[trigram_probs is None]
    bigram_probs = (bigram_probs, {"<meta_info>": {"count": 0, "total_tokens": 0}})[bigram_probs is None]
    unigram_probs = (unigram_probs, {"<meta_info>": {"count": 0, "total_tokens": 0}})[unigram_probs is None]

    for line in lines:
        for i in range(len(line) - 1):
            token1 = line[i]
            token2 = line[i + 1]
            if token1 not in bigram_probs:
                bigram_probs[token1] = {"<meta_info>": {"count": 0, "total_tokens": 0}}
                bigram_probs["<meta_info>"]["total_tokens"] += 1
            if token2 not in bigram_probs[token1]:
                bigram_probs[token1][token2] = 0
                bigram_probs[token1]["<meta_info>"]["total_tokens"] += 1
            bigram_probs[token1][token2] += 1
            bigram_probs[token1]["<meta_info>"]["count"] += 1
            bigram_probs["<meta_info>"]["count"] += 1

    for line in lines:
        for i in range(len(line)):
            token = line[i]
            if token not in unigram_probs:
                unigram_probs[token] = 0
                unigram_probs["<meta_info>"]["total_tokens"] += 1
            unigram_probs[token] += 1
            unigram_probs["<meta_info>"]["count"] += 1

    for line in lines:
        for i in range(len(line) - 2):
            token1 = line[i]
            token2 = line[i+1]
            token3 = line[i+2]
            if token1 not in trigram_probs:
                trigram_probs[token1] = {"<meta_info>": {"count": 0, "count_stat": [], "total_tokens": 0}}
                trigram_probs["<meta_info>"]["total_tokens"] += 1
            if token2 not in trigram_probs[token1]:
                trigram_probs[token1][token2] = {"<meta_info>": {"count": 0, "count_stat": [], "total_tokens": 0}}
                trigram_probs[token1]["<meta_info>"]["total_tokens"] += 1
            if token3 not in trigram_probs[token1][token2]:
                trigram_probs[token1][token2][token3] = 0
                trigram_probs[token1][token2]["<meta_info>"]["total_tokens"] += 1

            trigram_probs[token1][token2][token3] += 1
            trigram_probs[token1]["<meta_info>"]["count"] += 1
            trigram_probs[token1][token2]["<meta_info>"]["count"] += 1
            trigram_probs["<meta_info>"]["count"] += 1

    return unigram_probs, bigram_probs, trigram_probs

def laplace_unigram_prob(unigram, token):
    if token not in unigram:
        return 1 / (unigram["<meta_info>"]["count"] + unigram["<meta_info>"]["total_tokens"])
    return (unigram[token] + 1) / (unigram["<meta_info>"]["count"] + unigram["<meta_info>"]["total_tokens"])

def prob_with_interpolation_n_backoff_bi(tokens, unigram, bigram):
    l1 = 0.99
    l2 = 1 - l1

    if tokens[0] in bigram and tokens[1] in bigram[tokens[0]]:
        b_prob = (bigram[tokens[0]][tokens[1]] + 1) / (bigram[tokens[0]]["<meta_info>"]["count"] + bigram["<meta_info>"]["total_tokens"])
    else:
        if tokens[0] in bigram:
            cnt = bigram[tokens[0]]["<meta_info>"]["count"]
            tkn_cnt = bigram[tokens[0]]["<meta_info>"]["total_tokens"]
        else:
            cnt = bigram["<meta_info>"]["count"]
            tkn_cnt = bigram["<meta_info>"]["total_tokens"]

        b_prob = 1 / (cnt + tkn_cnt)

    if tokens[1] in unigram:
        u_prob = (unigram[tokens[1]] + 1) / (unigram["<meta_info>"]["count"] + unigram["<meta_info>"]["total_tokens"])
    else:
        u_prob = 1 / (unigram["<meta_info>"]["count"] + unigram["<meta_info>"]["total_tokens"])

    return u_prob * l2 + b_prob * l1

def prob_with_interpolation_n_backoff_tri(tokens, unigram, bigram, trigram):
    l1 = 0.9
    l2 = 0.099
    l3 = (1 - (l1 + l2))

    if tokens[0] in trigram and tokens[1] in trigram[tokens[0]] and tokens[2] in trigram[tokens[0]][tokens[1]]:
        t_prob = (trigram[tokens[0]][tokens[1]][tokens[2]] + 1) / (trigram[tokens[0]][tokens[1]]["<meta_info>"]["count"] + trigram[tokens[0]][tokens[1]]["<meta_info>"]["total_tokens"])
    else:
        if tokens[0] in trigram and tokens[1] in trigram[tokens[0]]:
            cnt = trigram[tokens[0]][tokens[1]]["<meta_info>"]["count"]
            tkn_cnt = trigram[tokens[0]][tokens[1]]["<meta_info>"]["total_tokens"]
        elif tokens[0] in trigram:
            cnt = trigram[tokens[0]]["<meta_info>"]["count"]
            tkn_cnt = trigram[tokens[0]]["<meta_info>"]["total_tokens"]
        else:
            cnt = trigram["<meta_info>"]["count"]
            tkn_cnt = trigram["<meta_info>"]["total_tokens"]

        t_prob = 1 / (cnt + tkn_cnt)

    if tokens[1] in bigram and tokens[2] in bigram[tokens[1]]:
        b_prob = (bigram[tokens[1]][tokens[2]] + 1) / (bigram[tokens[1]]["<meta_info>"]["count"] + bigram[tokens[1]]["<meta_info>"]["total_tokens"])
    else:
        if tokens[0] in bigram:
            cnt = bigram[tokens[0]]["<meta_info>"]["count"]
            tkn_cnt = bigram[tokens[0]]["<meta_info>"]["total_tokens"]
        else:
            cnt = bigram["<meta_info>"]["count"]
            tkn_cnt = bigram["<meta_info>"]["total_tokens"]
        b_prob = 1 / (cnt + tkn_cnt)

    if tokens[2] in unigram:
        u_prob = (unigram[tokens[2]] + 1) / (unigram["<meta_info>"]["count"] + unigram["<meta_info>"]["total_tokens"])
    else:
        u_prob = 1 / (unigram["<meta_info>"]["count"] + unigram["<meta_info>"]["total_tokens"])

    return t_prob * l1 + b_prob * l2 + u_prob * l3

def calculate_prob_stupid_off_tri(unigram, bigram, trigram, tokens, l=0.1):
    if tokens[0] not in trigram or tokens[1] not in trigram[tokens[0]] or tokens[2] not in trigram[tokens[0]][tokens[1]]:
        if tokens[1] not in bigram or tokens[2] not in bigram[tokens[1]]:
            return l**2 * laplace_unigram_prob(unigram, tokens[2])
        else:
            return l * bigram[tokens[1]][tokens[2]] / bigram[tokens[1]]["<meta_info>"]["count"]
    return trigram[tokens[0]][tokens[1]][tokens[2]] / trigram[tokens[0]][tokens[1]]["<meta_info>"]["count"]

def calculate_prob_stupid_off_bi(unigram, bigram, tokens, l=0.1):
    if tokens[0] not in bigram or tokens[1] not in bigram[tokens[0]]:
        return l * laplace_unigram_prob(unigram, tokens[1])
    return bigram[tokens[0]][tokens[1]] / bigram[tokens[0]]["<meta_info>"]["count"]

def calculate_perplexity(line, unigram, bigram, trigram, model = "stupid_backoff"):

    stupid_backoff_decay = 0.01

    if model == "stupid_backoff":
        prob = -math.log(calculate_prob_stupid_off_bi(unigram, bigram, line[0:2], l=stupid_backoff_decay))
    elif model == "interpolate_backoff":
        prob = -math.log(prob_with_interpolation_n_backoff_bi(line[0:2], unigram, bigram))
    else:
        raise NotImplementedError("model " + model + " not implemented")

    for i in range(len(line) - 2):
        if model == "stupid_backoff":
            prob -= math.log(calculate_prob_stupid_off_tri(unigram, bigram, trigram, line[i:i+3], l=stupid_backoff_decay))
        elif model == "interpolate_backoff":
            prob -= math.log(prob_with_interpolation_n_backoff_tri(line[i:i+3], unigram, bigram, trigram))

    # return 1 / math.pow(prob, 1 / len(line))
    return math.exp(prob / (len(line) - 1))

def part_a():
    unigram, bigram, trigram = create_trigram_probs(read_text("data/hw2-test.txt"))
    create_trigram_probs(read_text("data/hw2-train.txt"), trigram, bigram, unigram)
    print("Test + Train combined Trigrams: \n" + str(trigram))

    print("----------------------------")

    print("Only Train Trigrams: \n" + str(create_trigram_probs(read_text("data/hw2-train.txt"))[2]))

    print("----------------------------")

    print("Only Test Trigrams: \n" + str(create_trigram_probs(read_text("data/hw2-test.txt"))[2]))

def create_trigram_n_get_prplxity(model = "interpolate_backoff"):
    unigram, bigram, trigram = create_trigram_probs(read_text("data/hw2-train.txt"))
    test_text = read_text("data/hw2-test.txt")
    train_text = read_text("data/hw2-train.txt")

    print("--------------train perplexity--------------------------")
    for line in train_text:
        print(str(calculate_perplexity(line, unigram, bigram, trigram, model)) + " --  " + " ".join(line) + "\n")

    print("--------------test perplexity---------------------------")
    for line in test_text:
        print(str(calculate_perplexity(line, unigram, bigram, trigram, model)) + " --  " + " ".join(line) + "\n")
    print("--------------------------------------------------------")

print("---------------All trigrams---------------------")
part_a()
print("---------------End all trigrams-----------------")

create_trigram_n_get_prplxity("interpolate_backoff") #Options stupid_backoff | interpolate_backoff
