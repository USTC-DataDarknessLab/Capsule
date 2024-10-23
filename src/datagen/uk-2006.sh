#!/bin/bash
RAW_DATA_DIR='capsule/raw_dataset'
TW_RAW_DATA_DIR="${RAW_DATA_DIR}/uk-2006-05"
OUTPUT_DATA_DIR='capsule/dataset/uk-2006-05'

download(){
  mkdir -p ${UK_RAW_DATA_DIR}
  if [ ! -e "${UK_RAW_DATA_DIR}/uk-2006-05.graph" ]; then
    pushd ${UK_RAW_DATA_DIR}
    wget http://data.law.di.unimi.it/webdata/uk-2006-05/uk-2006-05.graph
    wget http://data.law.di.unimi.it/webdata/uk-2006-05/uk-2006-05.properties
    popd
  elif [ ! -e "${UK_RAW_DATA_DIR}/uk-2006-05.properties" ]; then
    pushd ${UK_RAW_DATA_DIR}
    wget http://data.law.di.unimi.it/webdata/uk-2006-05/uk-2006-05.properties
    popd
  else
    echo "Binary file already downloaded."
  fi
}

generate_coo(){
  download
  if [ ! -e "${UK_RAW_DATA_DIR}/coo.bin" ]; then
    java -cp ./utils/mavenWeb/target/webgraph-0.1-SNAPSHOT.jar it.unimi.dsi.webgraph.BVGraph -o -O -L "${UK_RAW_DATA_DIR}/uk-2006-05"
    java -cp ./utils/mavenWeb/target/webgraph-0.1-SNAPSHOT.jar ddl.sgg.WebgraphDecoder "${UK_RAW_DATA_DIR}/uk-2006-05"
    mv ${UK_RAW_DATA_DIR}/uk-2006-05_coo.bin ${UK_RAW_DATA_DIR}/coo.bin
  else
    echo "COO already generated."
  fi
}


generate_coo