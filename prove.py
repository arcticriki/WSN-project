from  lt_sampler import *

params =(10000, 0.05, 0.2)
degree_generator = PRNG(params)

[_, d, nums] = degree_generator.get_src_blocks()
