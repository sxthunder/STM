# STM(Statistic Text Matching)

STM is a tool for quickly text matching based on statistic features, such as tf-idf, bm25, edit-distance and so on. Nowdays a variety of deep learning methods have been applied to text matching, the importance of these statistic features could not be ignored, which could be used to build a baseline model or added as hand feature into deep neural network. Even though many packages support calculating these featues such as sklearn, it is frastrating that there is no framework which conbines these together. Thats why I wrote STM.

## What Can STM do:
### Calculate text similarity based on statistics
Supporting following similarities: (Multi-processing is used for spped)
1. Bow
2. Tf-idf
3. Bm25
4. Edit-distance
5. CQR
6. CTR

Support multi choices and combine in different ways.

### Search according the simialrity
For embedding based simialrity, Faiss is used for speed.