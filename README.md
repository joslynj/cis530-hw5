# cis530-hw5
Assignment writeup: http://computational-linguistics-class.org/homework/vector-semantics-2/vector-semantics-2.html

* [Pre-trained models in supported by Pymagnitude](https://github.com/plasticityai/magnitude)
* [sklearn.clusters library](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)

## Task 3.2 - Sparse Vectors ###
Test scores:
| Vector Model  | D=500, W=5 | D=500, W=3 | 
| ------------- | :-------------: | :-------------: |
| K-Means with PPMI | 0.387 |  |
| MiniBatch K-Means | 0.373 | 0.373 |
| MiniBatch K-Means  with PPMI | 0.405 | 0.386 |
| Agglomerative clustering with PPMI | 0.388 |  | 
| Spectral clustering with PPMI | 0.384 |  |

## Task 3.3 - Dense Vectors ###
### Test scores ###
| Vector Model  | fb_crawl | fb_wiki_subword + filtered_news | fb_wiki_subword + news | fb_crawl + filtered_news |
| ------------- | :-------------: | :-------------: | :-------------: | :-------------: |
| K-Means | 0.459  | 0.429 | 0.422 | |
| MiniBatch K-Means  | 0.456  | | | |
| Agglomerative clustering | 0.442| | | 0.433|
| Spectral clustering | 0.418 | | | |

### Leaderboard scores ###
Using agglomerative with linkage = "single" <3 with various vector models
* __fb_wiki_vectors + fb_crawl_vectors: 0.5295__ (our best model)
* fb_crawl_vectors + filtered_news_vectors: 0.5194
* fb_wiki_vectors + fb_crawl_vectors: 0.5124


### Vector Models ###
* fb_crawl: crawl-300d-2M.magnitude
* fb_wiki_subword: wiki-news-300d-1M-subword.magnitude
* fb_wiki: wiki-news-300d-1M.magnitude
* filter_news: GoogleNews-vectors-negative300.filter.magnitude
* news: GoogleNews-vectors-negative300.magnitude
* glove_crawl: glove.840B.300d.magnitude

## Task 3.4 - Clustering without K ##
### Dev set f-scores ###
| Vector Model  | fb_crawl | filtered_news | fb_wiki  | |
| ------------- | :-------------: | :-------------: | :-------------: | :-------------: |
| K-Means, k = [2,6] |  0.3894 |   |  |  |
| K-Means, k = [2,7] |  0.3944 |   |  |  |
| K-Means, k = [2,8] | 0.3706  |  0.3714 | 0.3711 |  |
| K-Means, k = [2,9] | 0.3639 |   |  |  |
| K-Means, k = [2,10] | 0.3652 |   |  |  |
| MiniBatch K-Means, k = [2,6]  |0.3955 |  | | |
| MiniBatch K-Means, k = [2,7]  | 0.3927 |  | | |
| MiniBatch K-Means, k = [2,8]  | 0.3773 | 0.3623 | 0.3669 | |
| MiniBatch K-Means, k = [2,9]  | 0.3816 |  | | |
| MiniBatch K-Means, k = [2,10] |0.3339 |   |  |  |
| Agglomerative clustering,  k = [2,8] | 0.3666 | | | |
| Spectral clustering,  k = [2,8] | 0.3597| | | |


### Leaderboard Scores ###
* KMeans, [2, 6] 0.4058
* KMeans, [2,7] 0.4148
* KMeans, [2, 6] 0.4058
* KMeans, [2, 8] 0.4216
* MiniBatch [2,6] 0.4163
* __MiniBatch [2,7] 0.4324__
* MiniBatch [2,8] 0.4132
