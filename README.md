# Capsule

Code repository for the SIGMOD 25 under review paper:
"Capsule: an Out-of-Core Training Mechanism for Colossal GNNs"

## Requirements

- python >= 3.8
- pytorch >= 1.12.0
- numpy >= 1.24.3
- dgl >= 0.9.1

Since our framework includes code based on DGL, you need to install a version of dgl >= 0.9.1 first. To prevent compatibility issues, it is recommended that users install the recommended version dgl 0.9.1-post. For specific installation methods, please refer to the official dgl website: https://docs.dgl.ai/en/0.9.x/install/index.html

## Prepare Datasets

We use six datasets in the paper: ogbn-papers, ogbn-products, Reddit,uk-2006-05, wb2001 and com_fr.

Users can download and process datasets according to the code in capsule/src/datagen. The operations for dataset processing here are referenced from GNNLab: https://github.com/SJTU-IPADS/gnnlab

## Preprocessing

This process generates subgraphs for the dataset and extracts features from the subgraphs, which can be implemented using the trans2subG.py provided in the code. Before performing the preprocessing of the dataset, you need to first download and process the dataset through the Prepare Datasets step, and you need to update the specific information of the dataset in capsule/datasetInfo.json:

```
"PA": {
        "edges": 1615685872,
        "nodes": 111059956,
        "rawFilePath": "capsule/data/raw/papers100M",
        "processedPath": "capsule/data/partition/PA",
        "featLen": 128,
        "classes": 172,
        "trainNUM": 1207179,
        "valNUM": 125265,
        "testNUM": 214338
    }
```

In the above, the rawFilePath and processedPath may need to be modified by the user according to their needs, while other data are the default configurations of the dataset and do not need to be modified. rawFilePath is the location of the original dataset, and processedPath is the output location for preprocessing operations.

After completing the above operations, you can complete the preprocessing operation with the following code. The processing can have two modes: one is the automatic partition merging method, and the other is the specified partition division method.:

```
# auto
python capsule/src/datapart/trans2subG.py --dataset=PD --cluster=4

# force
python capsule/src/datapart/trans2subG.py --dataset=PD --force_partiton --partNUM=4
```

## Train

Before training, users also need to edit the training configuration file in capsule/config according to the existing dataset. For example, after you have completed the preprocessing of the PD dataset, you can refer to the following example for training configuration:

```
{
    "train_name": "NC",
    "dataset": "PD",
    "model": "SAGE",
    "datasetpath": "./data/partition",
    "partNUM":4,
    "cacheNUM": 4,
    "batchsize": 1024,
    "maxEpoch": 20,
    "maxPartNodeNUM":10000000,
    "epochInterval": 5,
    "featlen": 100,
    "fanout": [
        10,
        10,
        10
    ],
    "classes": 47,
    "framework": "dgl",
    "mode": "train",
    "memUse": 1600000000,
    "GPUmem:" :1600000000,
    "edgecut" : 1,
    "nodecut" : 1,
    "featDevice" : "cuda:0" 
}
```

In the training configuration file, the configuration parameters that users need to pay attention to or modify are:

```
dataset: The name of the training dataset (consistent with datasetInfo.json)
model: The model used for training (we provide three default optional models: SAGE, GCN, GAT)
partNUM: The number of subgraphs (consistent with the results of preprocessing)
batchsize: Batch size
maxEpoch: The number of Epochs
epochInterval: Perform a test on the test set every how many Epochs
featlen: The size of the node features in the dataset (consistent with datasetInfo.json)
fanout: GNN layer configuration
classes: The number of output categories in the dataset (consistent with datasetInfo.json)
framework: Optional dgl or pyg
featDevice: Optional "cuda:0" or "cpu"
```

In addition to the above configurations, other configuration parameters can be the same as the default configuration provided.

After completing the above training configuration, you can start training with the following command:

```
python capsule/src/train/capsule/capsule_dgl_train.py --json_path="capsule/config/PD_dgl.json"
python capsule/src/train/capsule/capsule_pyg_train.py --json_path="capsule/config/PD_dgl.json"
```



---

## docker

If you need to build a Docker image and run code within it, you can follow these steps:

First, we need to build the image:

```sh
docker build -t name/capsule:v1.0 .
```

Next, we can start a container:

```sh
 docker run -d -it --gpus all -v /raw_file:/data images_name
```

Inside the container, we can execute the same steps as above. Note that **a custom version of DGL** is already installed in the container.

Run the code at the following path:

```sh
python capsule_dgl_train.py --json_path /data/config/PD_dgl.json
```

