#!/bin/bash

# ginex: https://github.com/SNU-ARC/Ginex


# First, we need to configure the environment for Ginex. In practice, 
# we believe that Capsule and Ginex can share the same runtime environment
python3 prepare_dataset.py



# When the training data has been transformed, we need to specify the cache parameters to generate the cached data files. 
# The setting of this size can have a significant impact on training.

# In general, we would try settings such as 45000000000, 6000000000, and 4500000000 
# to see which one is the fastest.
python3 create_neigh_cache.py --neigh-cache-size 6000000000


# Run directly
sudo PYTHONPATH=/home/bear/miniconda3/envs/graph/lib/python3.8/site-packages python -W ignore run_ginex.py --neigh-cache-size 6000000000 --feature-cache-size 6000000000
