#!/bin/bash
set -e

if [ $# -eq 0 ];
then
    echo
    echo "Usage:                    : $0 data_name alpha embed_size"
    echo
    echo "train_name                : required, name of abstracted train data"
    echo "test_name                 : required, name of abstracted test data"
    echo "method                    : required, classification method: knn/avg/mfs/nn"
    echo "option                    : optional, options"
    echo "mfs                       : optional, use most frequent sense for polysemy if train data is not available"
    echo "cos                       : optional, use cosine similarity for distance calculation"
    exit -1
fi

export THIS_DIR=$(cd $(dirname $0); pwd)
source ${THIS_DIR}/path.sh

train_name=$1
test_name=$2
method=$3
if ! [ -z $4 ]; then option="--option ${4}"; fi
if ! [ -z $5 ]; then mfs="--mfs"; fi
if ! [ -z $6 ]; then cos="--cos"; fi

train_path=${ABSTRACTED}/${train_name}
test_path=${ABSTRACTED}/${test_name}

${THIS_DIR}/wsd_classification.py ${train_path} ${test_path} ${method} ${option} ${mfs}
