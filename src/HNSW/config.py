"""Shared HNSW parameters used by candidate-edge organization."""

HNSW_SPACE = "cosine"  # 对于一般 embedding，L2 是一个常用选择
HNSW_M = 32  # 可以改成 32，看内存/速度情况
HNSW_EF_CONSTRUCTION = 200  # 一个比较通用的配置
HNSW_EF_SEARCH = 200  # 后续要查最近邻时可以进一步调
