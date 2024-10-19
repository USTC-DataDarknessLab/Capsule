# Introduction to Running the Baseline Code

## DGL

If the DGL environment is already installed, running the products, papers100M, or Reddit datasets provided directly by OGB/DGL should work without issues. However, if you want to run other datasets, you need to preprocess the data according to the capsule method first, converting the data into binary format.

The command to run can be referenced from `dgl/dgl.sh`. 



## PYG

If the PyG environment is already installed, running the products, papers100M, or Reddit datasets provided directly by OGB/DGL should work without issues. However, if you want to run other datasets, you need to preprocess the data according to the capsule method first, converting the data into binary format.

The command to run can be referenced from `pyg/pyg.sh`. 



## Ginex

Running Ginex can be done following the instructions in its [original repository](https://github.com/SNU-ARC/Ginex). Here, we also provide a modified version of the source code, with the main changes focused on the generation of the original dataset. We have made Ginex capable of directly generating the data needed for training from binary files.

The command to run can be referenced from `ginex/`. 



## Marius

For Marius, we mostly follow the training guidelines provided on its [original website](https://github.com/marius-team/marius). Our only modification involves calling the interface for custom datasets to connect our binary training files. After Marius was published, several updates were released, which introduced changes to its internal interfaces. We also found that running the disk logic often results in core dump issues. Therefore, during testing, we adhere to the principle of first attempting training with the disk version, and if an error occurs, switching to memory mode for training. We provide training files for PA under both modes. Additionally, we discovered that the number of partitions has a strong correlation with Marius's preprocessing and training.

The specific configuration files can be found in the directory `marius/`.