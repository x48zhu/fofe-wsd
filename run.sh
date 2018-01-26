#!/bin/bash
set -e

export THIS_DIR=$(cd $(dirname $0); pwd)
source ${THIS_DIR}/path.sh
source ${THIS_DIR}/config.sh

# Train Language Model
# Skip

LM_DIR=$1
MODEL_NAME=${LM_DIR}

# Save Language Model Parameters
${THIS_DIR}/torch-fofe/save.sh ${THIS_DIR}/torch-fofe-lm/ckpts/${LM_DIR}/fofe-train.model ${MODEL}/${MODEL_NAME}

# Abstract Context
#   MASC data type
for DATA_NAME in semcor masc
do
	echo "Abstract context for ${DATA_NAME}"
	${THIS_DIR}/context_abstract.sh ${MODEL_NAME} masc ${DATA_NAME} -a ${ALPHA} -n ${NGRAM} -d ${DESC}
done

${THIS_DIR}/classification.sh 2018-01-25-01-22-32_masc_semcor_trigram 2018-01-25-01-22-32_masc_masc_trigram knn 5

exit 0

#   OMSTI data type
for DATA_NAME in SemCor/semcor WSD_Unified_Evaluation_Datasets/senseval2 WSD_Unified_Evaluation_Datasets/semeval2013
do
	${THIS_DIR}/context_abstract.sh ${MODEL_NAME} omsti ${DATA_NAME} -a ${ALPHA} -n ${NGRAM} -d ${DESC}
done

${THIS_DIR}/classification.sh 2018-01-25-01-22-32_omsti_semcor_trigram 2018-01-25-01-22-32_omsti_senseval2_trigram knn 5

