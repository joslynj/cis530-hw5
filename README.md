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
Test scores:
| Vector Model  | fb_crawl | fb_wiki + filtered_news | fb_wiki + news | fb_crawl + filtered_news |
| ------------- | :-------------: | :-------------: | :-------------: | :-------------: |
| K-Means | 0.459  | 0.429 | 0.422 | |
| MiniBatch K-Means  | 0.456  | | | |
| Agglomerative clustering | 0.442| | | 0.433|
| Spectral clustering | 0.418 | | | |

* fb_crawl: crawl-300d-2M.magnitude
* fb_wiki_crawl: wiki-news-300d-1M-subword.magnitude
* filter_news: GoogleNews-vectors-negative300.filter.magnitude
* news: GoogleNews-vectors-negative300.magnitude
* glove_crawl:glove.840B.300d.magnitude
