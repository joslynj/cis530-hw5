import pandas as pd
import scipy.stats as stats
from pymagnitude import *


def main():
    vectors = Magnitude('vectors/GoogleNews-vectors-negative300.magnitude')
    df = pd.read_csv('data/SimLex-999.txt', sep='\t')[['word1', 'word2', 'SimLex999']]
    human_scores = []
    vector_scores = []
    human_low, human_high, vector_low, vector_high = float('inf'), float('-inf'), float('inf'), float('-inf')
    h_low_1, h_low_2, h_high_1, h_high_2, v_low_1, v_low_2, v_high_1, v_high_2 = '', '', '', '', '', '', '', ''

    for word1, word2, score in df.values.tolist():
        human_scores.append(score)
        similarity_score = vectors.similarity(word1, word2)
        vector_scores.append(similarity_score)
        print(f'{word1},{word2},{score},{similarity_score:.4f}')

        if score < human_low:
            human_low = score
            h_low_1, h_low_2 = word1, word2

        if similarity_score < vector_low:
            vector_low = similarity_score
            v_low_1, v_low_2 = word1, word2

        if score > human_high:
            human_high = score
            h_high_1, h_high_2 = word1, word2

        if similarity_score > vector_high:
            vector_high = similarity_score
            v_high_1, v_high_2 = word1, word2

    correlation, p_value = stats.kendalltau(human_scores, vector_scores)
    print(f'Correlation = {correlation}, P Value = {p_value}')

    print("human low score:", human_low)
    print("human low pair:", h_low_1, h_low_2)
    print("vector low score:", vector_low)
    print("vector low pair:", v_low_1, v_low_2)

    print("human high score:", human_high)
    print("human high pair:", h_high_1, h_high_2)
    print("vector high score:", vector_high)
    print("vector high pair:", v_high_1, v_high_2)

if __name__ == '__main__':
    main()
