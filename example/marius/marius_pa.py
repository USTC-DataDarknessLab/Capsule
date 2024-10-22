from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

import marius as m
from marius.tools.configuration.constants import PathConstants
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import NodeClassificationDataset
from marius.tools.preprocess.datasets.dataset_helpers import remap_nodes

class MYDATASET(NodeClassificationDataset):
    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "friendster"
        self.dataset_url = "https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz" # link to edges

    def download(self, overwrite=False):
        # These are the files we want to make my the end of the the download
        self.input_edge_list_file = self.output_directory / Path("graph.bin")
        self.input_node_feature_file = self.output_directory / Path("node-feat.csv")
        self.input_node_label_file = self.output_directory / Path("node-label.csv")
        self.input_train_nodes_file = self.output_directory / Path("train.csv")
        self.input_valid_nodes_file = self.output_directory / Path("valid.csv")
        self.input_test_nodes_file = self.output_directory / Path("test.csv")

        # If files already exist we don't need to do processing
        download = False
        if not self.input_edge_list_file.exists():
            download = True
        if not self.input_node_feature_file.exists():
            download = True
        if not self.input_node_label_file.exists():
            download = True
        if not self.input_train_nodes_file.exists():
            download = True
        if not self.input_valid_nodes_file.exists():
            download = True
        if not self.input_test_nodes_file.exists():
            download = True

        if download:
            print("stop and prepare files needed")

    def preprocess(self, num_partitions=16, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False):
        num_nodes = 111059956
        train_nodes = np.fromfile("./trainIds.bin",dtype=np.int64).astype(np.int32)
        valid_nodes = np.fromfile("./trainIds.bin",dtype=np.int64).astype(np.int32)
        test_nodes = np.fromfile("./trainIds.bin",dtype=np.int64).astype(np.int32)
        print('shape of train_nodes: ', train_nodes.shape)
        print('shape of valid_nodes: ', valid_nodes.shape)
        print('shape of test_nodes: ', test_nodes.shape)

        features = np.fromfile("./trainIds.bin",dtype=np.float32)
        print('shape of features: ', features.shape)
        labels = np.fromfile("./trainIds.bin",dtype=np.int32)
        print('shape of labels: ', labels.shape)

        train_edges = np.fromfile(self.input_edge_list_file,dtype=np.int32).reshape(-1,2)
        print('shape of train_edges: ', train_edges.shape)
        print(train_edges)
        # Calling the convert function to generate the preprocessed files
        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=train_edges,
            num_partitions=num_partitions,
            src_column=0,
            dst_column=1,
            remap_ids=remap_ids, # remap_ids here should be true
            sequential_train_nodes=sequential_train_nodes,
            format="numpy",
            known_node_ids=None, # modify if needed based on case A/B
            partitioned_evaluation=partitioned_eval,
        )
        dataset_stats = converter.convert()

        with open(self.train_nodes_file, "wb") as f:
            f.write(bytes(train_nodes))
        with open(self.valid_nodes_file, "wb") as f:
            f.write(bytes(valid_nodes))
        with open(self.test_nodes_file, "wb") as f:
            f.write(bytes(test_nodes))
        with open(self.node_features_file, "wb") as f:
            f.write(bytes(features))
        with open(self.node_labels_file, "wb") as f:
            f.write(bytes(labels))

        # update dataset yaml
        dataset_stats.num_train = train_nodes.shape[0]
        dataset_stats.num_valid = valid_nodes.shape[0]
        dataset_stats.num_test = test_nodes.shape[0]
        dataset_stats.node_feature_dim = features.shape[1]
        dataset_stats.num_classes = 172

        dataset_stats.num_nodes = num_nodes

        with open(self.output_directory / Path("dataset.yaml"), "w") as f:
            yaml_file = OmegaConf.to_yaml(dataset_stats)
            f.writelines(yaml_file)

        return dataset_stats

if __name__ == "__main__":
    # initialize and preprocess dataset
    dataset_dir = Path(".") #output dir
    dataset = MYDATASET(dataset_dir)
    if not (dataset_dir / Path("edges/train_edges.bin")).exists():
        dataset.download()
        dataset.preprocess()

    dataset_stats = OmegaConf.load(dataset_dir / Path("dataset.yaml"))

