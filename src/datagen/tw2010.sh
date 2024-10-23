#!/bin/bash
RAW_DATA_DIR='capsule/raw_dataset'
TW_RAW_DATA_DIR="${RAW_DATA_DIR}/twitter"
OUTPUT_DATA_DIR='capsule/dataset/twitter'
# download raw dataset

download(){
  mkdir -p ${TW_RAW_DATA_DIR}
  if [ ! -e "${TW_RAW_DATA_DIR}/twitter-2010.graph" ]; then
    pushd ${TW_RAW_DATA_DIR}
    wget http://data.law.di.unimi.it/webdata/twitter-2010/twitter-2010.graph
    wget http://data.law.di.unimi.it/webdata/twitter-2010/twitter-2010.properties
    popd
  elif [ ! -e "${TW_RAW_DATA_DIR}/twitter-2010.properties" ]; then
    pushd ${TW_RAW_DATA_DIR}
    wget http://data.law.di.unimi.it/webdata/twitter-2010/twitter-2010.properties
    popd
  else
    echo "Binary file already downloaded."
  fi
}

generate_coo(){
  download
  if [ ! -e "${TW_RAW_DATA_DIR}/coo.bin" ]; then
    java -cp ./utils/mavenWeb/target/webgraph-0.1-SNAPSHOT.jar it.unimi.dsi.webgraph.BVGraph -o -O -L "${TW_RAW_DATA_DIR}/twitter-2010"
    java -cp ./utils/mavenWeb/target/webgraph-0.1-SNAPSHOT.jar ddl.sgg.WebgraphDecoder "${TW_RAW_DATA_DIR}/twitter-2010"
    mv ${TW_RAW_DATA_DIR}/twitter-2010_coo.bin ${TW_RAW_DATA_DIR}/coo.bin
  else
    echo "COO already generated."
  fi
}


generate_coo