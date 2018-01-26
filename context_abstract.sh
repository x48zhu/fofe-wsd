#!/bin/bash
set -e

if [ $# -eq 0 ];
then
    echo
    echo "Usage:            : $0 data_name alpha embed_size"
    echo
    echo "model_name        : required, name of the pre-trained language model"
    echo "data_type         : required, type of the data (masc/omsti/noad)"
    echo "data_name         : required, name of the data to abstract"
    echo "alpha             : required, forgeting factor"
    echo "description       : optional, extra info add to output file name to distinguish"
    exit -1;
fi

model_name=$1
data_type=$2
data_name=$3
shift; shift; shift

while getopts "a:d:n:p" opt; do
    case "$opt" in
        a) alpha="--alpha ${OPTARG}" ;;
	d) description="--desc ${OPTARG}" ;;
        n) ngram="--ngram ${OPTARG}" ;;
	p) processed="--processed" ;;
    esac
done

export THIS_DIR=$(cd $(dirname $0); pwd)
source ${THIS_DIR}/path.sh

model_path=${MODEL}/${model_name}
data_path=${DATA}/${data_type}/${data_name}
wordlist_path=${THIS_DIR}/torch-fofe/numeric-data/google-100000.vocab
output_dir=${ABSTRACTED}/

${THIS_DIR}/context_abstract.py ${model_path} ${wordlist_path} ${data_type} ${data_path} ${output_dir} ${description} ${alpha} ${processed} ${ngram}

