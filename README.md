# STM(Statistic Text Matching)


[![standard-readme compliant](https://img.shields.io/badge/Hello-Language-brightgreen](https://github.com/sxthunder/STM/README-en.md)

STM(Statistic Text Matching)：是一个基于统计特征进行文本匹配的工具。深度学习如火如荼，但是此等基于统计的特征仍然在实际应用中举足轻重。一些常见的统计方法例如：Tf-idf、Bm-25、各种距离以及ctq、cqr分散在不同的包中，每次使用都要读不同包的文档，如果想同时使用几种统计量会写大量重复代码，本框架旨在简化这一部分的工作量。

## STM能做什么:
### 计算不同的统计相似度特征
支持一下几种类型的相似度计算（为了提高效率会使用多线程）
1. Bow
2. Tf-idf
3. Bm25
4. Edit-distance
5. CQR
6. CTR

支持同时计算多种相似度，并采用多种方式进行融合。

### 基于相似度进行匹配搜索
基于embedding的相似度，引入Faiss进行提速。