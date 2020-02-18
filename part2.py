import pandas as pd
import scipy.stats as stats
from pymagnitude import *


def main():
    vectors = Magnitude('vectors/GoogleNews-vectors-negative300.magnitude')
    df = pd.read_csv('data/SimLex-999.txt', sep='\t')[['word1', 'word2', 'SimLex999']]
    human_scores = []
    vector_scores = []
    file = open('output.txt', 'w')
    for word1, word2, score in df.values.tolist():
        human_scores.append(score)
        similarity_score = vectors.similarity(word1, word2)
        vector_scores.append(similarity_score)
        file.write(f'{word1},{word2},{score},{similarity_score:.4f}\n')

    correlation, p_value = stats.kendalltau(human_scores, vector_scores)
    file.write(f'Correlation = {correlation}, P Value = {p_value}\n')
    file.close()


if __name__ == '__main__':
    main()
