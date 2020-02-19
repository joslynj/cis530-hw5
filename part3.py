from pymagnitude import *
from itertools import combinations
from prettytable import PrettyTable
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import random


def load_input_file(file_path):
    """
    Loads the input file to two dictionaries
    :param file_path: path to an input file
    :return: 2 dictionaries:
    1. Dictionary, where key is a target word and value is a list of paraphrases
    2. Dictionary, where key is a target word and value is a number of clusters
    """
    word_to_paraphrases_dict = {}
    word_to_k_dict = {}

    with open(file_path, 'r') as fin:
        for line in fin:
            target_word, k, paraphrases = line.split(' :: ')
            word_to_k_dict[target_word] = int(k)
            word_to_paraphrases_dict[target_word] = paraphrases.split()

    return word_to_paraphrases_dict, word_to_k_dict


def load_output_file(file_path):
    """
    :param file_path: path to an output file
    :return: A dictionary, where key is a target word and value is a list of list of paraphrases
    """
    clusterings = {}

    with open(file_path, 'r') as fin:
        for line in fin:
            target_word, _, paraphrases_in_cluster = line.strip().split(' :: ')
            paraphrases_list = paraphrases_in_cluster.strip().split()
            if target_word not in clusterings:
                clusterings[target_word] = []
            clusterings[target_word].append(paraphrases_list)

    return clusterings


def write_to_output_file(file_path, clusterings):
    """
    Writes the result of clusterings into an output file
    :param file_path: path to an output file
    :param clusterings:  A dictionary, where key is a target word and value is a list of list of paraphrases
    :return: N/A
    """
    with open(file_path, 'w') as fout:
        for target_word, clustering in clusterings.items():
            for i, cluster in enumerate(clustering):
                fout.write(f'{target_word} :: {i + 1} :: {" ".join(cluster)}\n')
        fout.close()


def get_paired_f_score(gold_clustering, predicted_clustering):
    """
    :param gold_clustering: gold list of list of paraphrases
    :param predicted_clustering: predicted list of list of paraphrases
    :return: Paired F-Score
    """
    gold_pairs = set()
    for gold_cluster in gold_clustering:
        for pair in combinations(gold_cluster, 2):
            gold_pairs.add(tuple(sorted(pair)))

    predicted_pairs = set()
    for predicted_cluster in predicted_clustering:
        for pair in combinations(predicted_cluster, 2):
            predicted_pairs.add(tuple(sorted(pair)))

    overlapping_pairs = gold_pairs & predicted_pairs

    precision = 1. if len(predicted_pairs) == 0 else float(len(overlapping_pairs)) / len(predicted_pairs)
    recall = 1. if len(gold_pairs) == 0 else float(len(overlapping_pairs)) / len(gold_pairs)
    paired_f_score = 0. if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return paired_f_score


def evaluate_clusterings(gold_clusterings, predicted_clusterings):
    """
    Displays evaluation scores between gold and predicted clusterings
    :param gold_clusterings: dictionary where key is a target word and value is a list of list of paraphrases
    :param predicted_clusterings: dictionary where key is a target word and value is a list of list of paraphrases
    :return: N/A
    """
    target_words = set(gold_clusterings.keys()) & set(predicted_clusterings.keys())

    if len(target_words) == 0:
        print('No overlapping target words in ground-truth and predicted files')
        return None

    paired_f_scores = np.zeros((len(target_words)))
    ks = np.zeros((len(target_words)))

    table = PrettyTable(['Target', 'k', 'Paired F-Score'])
    for i, target_word in enumerate(target_words):
        paired_f_score = get_paired_f_score(gold_clusterings[target_word], predicted_clusterings[target_word])
        k = len(gold_clusterings[target_word])
        paired_f_scores[i] = paired_f_score
        ks[i] = k
        table.add_row([target_word, k, f'{paired_f_score:0.4f}'])

    average_f_score = np.average(paired_f_scores, weights=ks)
    print(table)
    print(f'=> Average Paired F-Score:  {average_f_score:.4f}')

def create_PPMI_matrix(term_context_matrix):

    num_terms = term_context_matrix.shape[0]
    e = 1e-6 # smoothing factor to prevent divison-by-zero
    total = np.sum(term_context_matrix + e)
    numerator = (term_context_matrix+  e) / total
    context_totals = np.sum(numerator, axis=0) # total context counts (sum columns)
    word_totals = np.sum(numerator, axis=1) # total word counts (sum rows)
    
    denominator = np.outer(context_totals, word_totals)
    pmi_matrix = np.divide(numerator, denominator)
    pmi_matrix = np.log2(pmi_matrix)
    ppmi_matrix = np.maximum(pmi_matrix, 0.)
    
    return ppmi_matrix

# TASK 3.1
def cluster_random(word_to_paraphrases_dict, word_to_k_dict):
    """
    Clusters paraphrases randomly
    :param word_to_paraphrases_dict: dictionary, where key is a target word and value is a list of paraphrases
    :param word_to_k_dict: dictionary, where key is a target word and value is a number of clusters
    :return: dictionary, where key is a target word and value is a list of list of paraphrases,
    where each list corresponds to a cluster
    """
    clusterings = {}
    random.seed(123)

    for target_word in word_to_paraphrases_dict.keys():
        paraphrase_list = word_to_paraphrases_dict[target_word]
        k = word_to_k_dict[target_word]
        # TODO: Implement
        paraphrase_list_check = paraphrase_list.copy()
        list_of_paraphrases = []
        i = 0
        while len(paraphrase_list_check) != 0:
            if i == k:
                i = 0 
            word = random.choice(paraphrase_list)
            if word in paraphrase_list_check:
                paraphrase_list_check.remove(word)
            if len(list_of_paraphrases) < i + 1:
                lst = []
                lst.append(word)
                list_of_paraphrases.append(lst)
            else:
                if word not in list_of_paraphrases[i]:
                    list_of_paraphrases[i].append(word)
            i += 1

        clusterings[target_word] = list_of_paraphrases

    return clusterings

# TASK 3.2
def cluster_with_sparse_representation(word_to_paraphrases_dict, word_to_k_dict):
    """
    Clusters paraphrases using sparse vector representation
    :param word_to_paraphrases_dict: dictionary, where key is a target word and value is a list of paraphrases
    :param word_to_k_dict: dictionary, where key is a target word and value is a number of clusters
    :return: dictionary, where key is a target word and value is a list of list of paraphrases,
    where each list corresponds to a cluster
    """
    # Note: any vector representation should be in the same directory as this file
    vectors = Magnitude("vectors/coocvec-500mostfreq-window-3.filter.magnitude")
    clusterings = {}

    for target_word in word_to_paraphrases_dict.keys():
        paraphrase_list = word_to_paraphrases_dict[target_word]
        k = word_to_k_dict[target_word]
        # TODO: Implement
        list_of_paraphrases = []
        num_words = len(paraphrase_list)
        vectors_list = vectors.query(paraphrase_list)

        ### Experiment with removing all-zero columns/dimensions ###
        # np_vectors = np.array(vectors_list)
        # columns = np.argwhere(np.all(np_vectors[..., :] == 0, axis=0))
        # vectors_list = np.delete(np_vectors, columns, axis=1)

        ### Experiment with different clustering algorithms ###
        # clusters = KMeans(n_clusters=k).fit(vectors_list) # 0.2555
        clusters = MiniBatchKMeans(n_clusters=k).fit(vectors_list) # 0.2662
        # clusters = SpectralClustering(n_clusters=k).fit(vectors_list) # 0.2351
        # clusters = AgglomerativeClustering(n_clusters=k).fit(vectors_list) # 0.2640
        
        for i in range(k): # for each cluster
            lst = []
            for word_ind in range(num_words):
                if clusters.labels_[word_ind] == i: # if label is i (word belongs to ith cluster)
                    lst.append(paraphrase_list[word_ind])
            list_of_paraphrases.append(lst)
        clusterings[target_word] = list_of_paraphrases

    return clusterings


# TASK 3.3
def cluster_with_dense_representation(word_to_paraphrases_dict, word_to_k_dict):
    """
    Clusters paraphrases using dense vector representation
    :param word_to_paraphrases_dict: dictionary, where key is a target word and value is a list of paraphrases
    :param word_to_k_dict: dictionary, where key is a target word and value is a number of clusters
    :return: dictionary, where key is a target word and value is a list of list of paraphrases,
    where each list corresponds to a cluster
    """
    # Note: any vector representation should be in the same directory as this file
    google_vectors = Magnitude("vectors/GoogleNews-vectors-negative300.filter.magnitude") # KMeans: 0.3225
    # vectors = Magnitude("vectors/GoogleNews-vectors-negative300.magnitude") # KMeans: 0.3017 AG: 0.3084
    wiki_vectors = Magnitude("vectors/wiki-news-300d-1M-subword.magnitude") # KMeans: 0.3302 AG: 0.3249 MiniBatch: 0.3313
    crawl_vectors = Magnitude("vectors/crawl-300d-2M.magnitude") # KMeans: 0.3508 AG: 0.3256 MiniBatch: 0.3427
    # vectors = Magnitude("vectors/glove.twitter.27B.25d.magnitude") # KMeans: 0.2327 AG: 0.2340 MiniBatch: 0.2424
    # glove_vectors = Magnitude("vectors/glove-lemmatized.6B.300d.magnitude") # MiniBatch: 0.2739 AG: 0.2659
    # glove_vectors = Magnitude("vectors/glove.6B.300d.magnitude")
    # twitter_vectors = Magnitude("vectors/glove.twitter.27B.200d.magnitude") # MiniBatch: 0.2623
    # vectors = Magnitude(crawl_vectors, twitter_vectors) # KMeans: 0.3017
    # vectors = Magnitude(crawl_vectors, google_vectors) # KMeans: 0.3315
    # vectors = Magnitude(crawl_vectors, glove_vectors) # KMeans: 0.3026
    # vectors = Magnitude(wiki_vectors, twitter_vectors) # KMeans: 0.2887
    vectors = Magnitude(wiki_vectors, google_vectors) # 0.3290

    clusterings = {}

    for target_word in word_to_paraphrases_dict.keys():
        paraphrase_list = word_to_paraphrases_dict[target_word]
        k = word_to_k_dict[target_word]
        # TODO: Implement
        vectors_list = vectors.query(paraphrase_list)

        # Create k clusters
        # clusters = AgglomerativeClustering(n_clusters=k).fit(vectors_list)
        # clusters = MiniBatchKMeans(n_clusters=k).fit(vectors_list)
        # clusters = SpectralClustering(n_clusters=k).fit(vectors_list)
        clusters = KMeans(n_clusters=k).fit(vectors_list)
        cluster_dict = {i: np.where(clusters.labels_ == i)[0] for i in range(clusters.n_clusters)}
        list_of_paraphrases = []
        for key, value in cluster_dict.items():
            paraphrases = []
            for i in value:
                paraphrases.append(paraphrase_list[i])
            list_of_paraphrases.append(paraphrases)

        clusterings[target_word] = list_of_paraphrases

    return clusterings


# TASK 3.4
def cluster_with_no_k(word_to_paraphrases_dict):
    """
    Clusters paraphrases using any vector representation
    :param word_to_paraphrases_dict: dictionary, where key is a target word and value is a list of paraphrases
    :return: dictionary, where key is a target word and value is a list of list of paraphrases,
    where each list corresponds to a cluster
    """
    # Note: any vector representation should be in the same directory as this file
    vectors = Magnitude("vectors/GoogleNews-vectors-negative300.filter.magnitude")
    clusterings = {}

    for target_word in word_to_paraphrases_dict.keys():
        paraphrase_list = word_to_paraphrases_dict[target_word]
        # TODO: Implement
        clusterings[target_word] = None

    return clusterings

def main():
    # pass

    ##### Task 3.1: Cluster Randomly #####
    # word_to_paraphrases_dict, word_to_k_dict = load_input_file('data/dev_input.txt')
    # gold_clusterings = load_output_file('data/dev_output.txt')
    # predicted_clusterings = cluster_random(word_to_paraphrases_dict, word_to_k_dict)
    # evaluate_clusterings(gold_clusterings, predicted_clusterings)

    # word_to_paraphrases_dict, word_to_k_dict = load_input_file('data/test_input.txt')
    # predicted_clusterings = cluster_random(word_to_paraphrases_dict, word_to_k_dict)
    # write_to_output_file('test_output_random.txt', predicted_clusterings)

    ##### Task 3.2 Cluster with Sparse Representations #####
    # word_to_paraphrases_dict, word_to_k_dict = load_input_file('data/dev_input.txt')
    # gold_clusterings = load_output_file('data/dev_output.txt')
    # predicted_clusterings = cluster_with_sparse_representation(word_to_paraphrases_dict, word_to_k_dict)
    # evaluate_clusterings(gold_clusterings, predicted_clusterings)

    # word_to_paraphrases_dict, word_to_k_dict = load_input_file('data/test_input.txt')
    # predicted_clusterings = cluster_with_sparse_representation(word_to_paraphrases_dict, word_to_k_dict)
    # write_to_output_file('test_output_sparse.txt', predicted_clusterings)

    ##### Task 3.3 Cluster with Dense Representations #####
    # word_to_paraphrases_dict, word_to_k_dict = load_input_file('data/dev_input.txt')
    # gold_clusterings = load_output_file('data/dev_output.txt')
    # predicted_clusterings = cluster_with_dense_representation(word_to_paraphrases_dict, word_to_k_dict)
    # evaluate_clusterings(gold_clusterings, predicted_clusterings)

    # word_to_paraphrases_dict, word_to_k_dict = load_input_file('data/test_input.txt')
    # predicted_clusterings = cluster_with_dense_representation(word_to_paraphrases_dict, word_to_k_dict)
    # write_to_output_file('test_output_dense.txt', predicted_clusterings)

    ##### Task 3.4 Cluster withour K #####
    '''
    word_to_paraphrases_dict, _ = load_input_file('data/dev_input.txt')
    gold_clusterings = load_output_file('data/dev_output.txt')
    predicted_clusterings = cluster_with_no_k(word_to_paraphrases_dict)
    valuate_clusterings(gold_clusterings, predicted_clusterings)

    word_to_paraphrases_dict, _ = load_input_file('data/test_nok_input.txt')
    predicted_clusterings = cluster_with_no_k(word_to_paraphrases_dict)
    write_to_output_file('test_nok_output.txt', predicted_clusterings)
    '''
    

if __name__ == '__main__':
    main()