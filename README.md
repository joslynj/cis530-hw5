# cis530-hw5
Assignment writeup: http://computational-linguistics-class.org/homework/vector-semantics-2/vector-semantics-2.html

* [Pre-trained models in supported by Pymagnitude](https://github.com/plasticityai/magnitude)
* [sklearn.clusters library](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)

## Task 3.2 - Sparse Vectors ###
Test scores:
| Vector Model  | D=500, W=5 | D=500, W=3 | 
| ------------- | ------------- | ------------- |
| K-Means with PPMI | 0.387 | N/A |
| MiniBatch K-Means | 0.373 | 0.373 |
| MiniBatch K-Means  with PPMI | 0.405 | 0.386 |
| Agglomerative clustering with PPMI | 0.388 | N/A | 
| Spectral clustering with PPMI | 0.384 | N/A |

## Task 3.3 - Dense Vectors ###
Test scores:
| Vector Model  | fb_crawl | wiki_vectors + news_filtered | wiki_vectors + news_vectors | fb_crawl + news_filtered |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| K-Means | 0.459  | 0.429 | 0.422 | |
| MiniBatch K-Means  | 0.456  | | | |
| Agglomerative clustering | 0.442| | | 0.433|
| Spectral clustering | 0.418 | | | |
