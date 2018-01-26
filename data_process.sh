#!/bin/bash
set -e

if [ $# -eq 0 ];
then
    echo
    echo "Usage:                    : $0 data_name alpha embed_size"
    echo
    echo "data_type                 : required, type of the data (masc/omsti/noad)"
    echo "data_name                 : required, name of the data to abstract"
    exit -1;
fi

export THIS_DIR=$(cd $(dirname $0); pwd)
source ${THIS_DIR}/path.sh

data_type=$1
data_name=$2

data_path=${DATA}/${data_type}/${data_name}
wordlist_path=${THIS_DIR}/torch-fofe/numerica-data/google-100000.vocab
output_dir=${ABSTRACTED}/

./fofe-wsd/data_process.py ${wordlist_path} ${data_type} ${data_path} ${output_dir}

