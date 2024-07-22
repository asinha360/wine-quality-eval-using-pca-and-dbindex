# Wine Quality Evaluation using PCA through SVD

This project explores the use of PCA to reduce dimensionality and evaluate k-means clustering generated by
applying it on raw and standardized data using DB index scores.

PCA is a dimensionality reduction technique used to explore and visualize high- dimensional data in a low dimensional space whilst retaining the most important information in the dataset by computing principal components that define most of the variance present in the data. 

This was done with the aid of SVD which provided the right singular vector V, which when multiplied by a raw or standardized independent variable matrix produces a ranking of the principal components involved in the dataset.

DB index index is, in essence, a clustering evaluation metric which is widely used to assess the quality of a clusters produced by any specialized algorithm.

The question being explored is the effectiveness of a PCA algorithm applied to raw and standardized datasets in conjunction with a DB index evaluatory metric and its use in actually finding principal components in contrast to doing so with SVD.

Read a3.pdf for a deeper look.