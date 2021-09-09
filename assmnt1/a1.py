import math
import sys

# This boilerplate provides a rough structure for your code You are free to change it
# as you wish. You are in fact encouraged to do so, with the exception of the main
# function. Please do not change the main function.

def clean_text(input_text, include_space = False):
    """Cleans the text according to the problem description"""
    input_text = input_text.lower()
    input_text = input_text.strip()
    input_text = "".join([ch for i, ch in enumerate(input_text) if ch.isalpha() or (include_space and ch == " " and i>0 and input_text[i-1] != " ")])
    return input_text

def learn_distribution(input_text, include_space = False):
    """Learn probability distribution from text file.
        Remember to call clean
    """
    text = clean_text(input_text, include_space)
    letter_freq = {chr(i + 97): 0 for i in range(26)}
    if include_space:
        letter_freq[" "] = 0

    total_chars = 0

    for ch in text:
        total_chars += 1
        letter_freq[ch] += 1

    char_prob = {k: letter_freq[k] / total_chars for k in letter_freq}

    return char_prob

def learn_distribution_from_file(input_file_name):
    """Call learn distribution here after reading the file"""
    with open(input_file_name, "r") as f:
        text = "".join(f.readlines())
    return learn_distribution(text, include_space=True)

def expectation(distribution):
    """Return expectation of distribution"""
    return sum([k * distribution[k] for k in distribution])


def variance(distribution):
    """Return variance of distribution"""
    expected_val_sq = sum([distribution[k] * (k ** 2) for k in distribution])
    return expected_val_sq - expectation(distribution) ** 2


def entropy(distribution):
    """Return entropy of distribution"""
    total_ent = 0
    for k in distribution:
        if distribution[k] == 0:
            continue
        total_ent += -distribution[k] * math.log2(distribution[k])
    return total_ent

def get_min_2(dist):
    keys = list(dist.keys())
    min_2 = ([keys[0], keys[1]], [keys[1], keys[0]])[dist[keys[0]]["total_p"] > dist[keys[1]]["total_p"]]

    for i in range(2, len(keys)):
        if dist[keys[i]]["total_p"] < dist[min_2[0]]["total_p"]:
            min_2[1] = min_2[0]
            min_2[0] = keys[i]
        elif dist[keys[i]]["total_p"] < dist[min_2[1]]["total_p"]:
            min_2[1] = keys[i]
    return min_2

def get_new_encoding(dist, min_2):
    min1_code = {k: "0" + dist[min_2[0]]["encoding"][k] for k in dist[min_2[0]]["encoding"]}
    min2_code = {k: "1" + dist[min_2[1]]["encoding"][k] for k in dist[min_2[1]]["encoding"]}
    min1_code.update(min2_code)
    dist["".join(min_2)] = {"encoding": min1_code, "total_p": dist[min_2[0]]["total_p"] + dist[min_2[1]]["total_p"]}
    dist.pop(min_2[0], None)
    dist.pop(min_2[1], None)

def huffman_encoding(dist):
    new_dist = {k:{"encoding": {k:''}, "total_p": dist[k]} for k in dist}
    while(len(new_dist.keys()) > 1):
        min_2 = get_min_2(new_dist)
        get_new_encoding(new_dist, min_2)
    return new_dist["".join(min_2)]["encoding"]

def get_encoding_len_dist(encoding, dist):
    encoding_len = {k: len(encoding[k]) for k in encoding}
    len_dist = {}
    for k in encoding_len:
        if encoding_len[k] not in len_dist:
            len_dist[encoding_len[k]] = 0
        len_dist[encoding_len[k]] += dist[k]

    return len_dist

def p1():
    """This function should print the expected value, variance and entropy"""
    text = """Bach was the most famous composer in the world, and so wasHandel.   Handel  was  half  
        German, half  Italian  and  half  En-glish.  He was very large.  Bach died from 1750 to the 
        present. Beethoven  wrote  music  even  though  he  was  deaf.   He  was  sodeaf he wrote 
        loud music. He took long walks in the forest evenwhen everyone was calling for him.  Beethoven expired 
        in 1827and later died for this."""

    char_prob = learn_distribution(text)

    print("Probability of individual chars: " + str(char_prob))

    print("Total digits needed (Entropy): " + str(entropy(char_prob)))

    efficient_encoding = huffman_encoding(char_prob)

    distribution = get_encoding_len_dist(efficient_encoding, char_prob)

    expected_val = expectation(distribution)

    print("Efficient encoding: " + str(efficient_encoding))

    print("Expected value of efficient encoding: " + str(expected_val))

    print("Variance of the code: " + str(variance(distribution)))

    return char_prob

def replace_zero_dist(distribution):
    new_dist = {k: (distribution[k], 0.0001)[distribution[k] == 0] for k in distribution}
    total_val = sum(new_dist.values())
    return {k: new_dist[k]/total_val for k in new_dist}

def get_kl_div(dist1, dist2):
    return sum([dist1[k] * (math.log2(dist1[k]) - math.log2(dist2[k])) for k in dist1])

def p2():
    """This problem should print the KL divergence of the two distributions in problem
    2"""
    text1 = """Bach was the most famous composer in the world, and so wasHandel.   Handel  was  half  
                German, half  Italian  and  half  En-glish.  He was very large.  Bach died from 1750 to the 
                present. Beethoven  wrote  music  even  though  he  was  deaf.   He  was  sodeaf he wrote 
                loud music. He took long walks in the forest evenwhen everyone was calling for him.  Beethoven expired 
                in 1827and later died for this."""

    text2 = """The  sun  never  set  on  the  British  Empire  because  the  BritishEmpire
                 is in the East and the sun sets in the West.  Queen Vic-toria was the longest queen.  
                 She sat on a thorn for 63 years.He reclining years and finally the end of her life 
                 were exempla-tory of a great personality.  Her death was the final event whichended 
                 her reign."""

    dist1 = replace_zero_dist(learn_distribution(text1))
    dist2 = replace_zero_dist(learn_distribution(text2))

    print("KL divergence (1 || 2): " + str(get_kl_div(dist1, dist2)))
    print("KL divergence (2 || 1): " + str(get_kl_div(dist2, dist1)))


def p3(input_file_name):
    """This function should read the contents of the input file_name and identify the
    language of the text contained in it and print the name of the language"""
    data_dir = "data/"
    file1 = data_dir + "english.txt"
    sample_file1 = data_dir + "english_sample.txt"
    file2 = data_dir + "polish.txt"
    sample_file2 = data_dir + "polish_sample.txt"

    dist1 = replace_zero_dist(learn_distribution_from_file(file1))
    dist2 = replace_zero_dist(learn_distribution_from_file(file2))
    dist1_sample = replace_zero_dist(learn_distribution_from_file(sample_file1))
    dist2_sample = replace_zero_dist(learn_distribution_from_file(sample_file2))

    print("------Part 3.a)------")
    print("Entropy " + file1 + ": " + str(entropy(dist1)))
    print("Entropy " + file2 + ": " + str(entropy(dist2)))
    print("Entropy " + sample_file1 + ": " + str(entropy(dist1_sample)))
    print("Entropy " + sample_file2 + ": " + str(entropy(dist2_sample)))

    print("\n\n------part 3.b)------")
    print("KL Divergence " + file1 + " || " + sample_file1 + " : " + str(get_kl_div(dist1, dist1_sample)))
    print("KL Divergence " + file1 + " || " + sample_file2 + " : " + str(get_kl_div(dist1, dist2_sample)))
    print("KL Divergence " + file2 + " || " + sample_file1 + " : " + str(get_kl_div(dist2, dist1_sample)))
    print("KL Divergence " + file2 + " || " + sample_file2 + " : " + str(get_kl_div(dist2, dist2_sample)))

if __name__ == "__main__":
    if sys.argv[1] == '1':
        p1()
    elif sys.argv[1] == '2':
        p2()
    else:
        p3(sys.argv[1])
