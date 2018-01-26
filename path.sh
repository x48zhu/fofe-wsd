#!/bin/bash

export THIS_DIR=/eecs/research/asr/xzhu/work/fofe-wsd
export DATA=${THIS_DIR}/data
export MODEL=${THIS_DIR}/model
export ABSTRACTED=${THIS_DIR}/abstracted

[ -d ${MODEL} ] || mkdir -p ${MODEL}
[ -d ${ABSTRACTED} ] || mkdir -p ${ABSTRACTED}

export KNRM="\x1B[0m"
export KRED="\x1B[31m"
export KGRN="\x1B[32m"
export KYEL="\x1B[33m"
export KBLU="\x1B[34m"
export KMAG="\x1B[35m"
export KCYN="\x1B[36m"
export KWHT="\x1B[37m"

function INFO() {
    msg="$@"
    printf "${KGRN}"
    printf "`date +"%Y-%m-%d %H-%M-%S"` [INFO]: ${msg}\n"
    printf "${KNRM}"
}

function CRITICAL() {
    msg="$@"
    printf "${KRED}"
    printf "`date +"%Y-%m-%d %H-%M-%S"` [CRITICAL]: ${msg}\n"
    printf "${KNRM}"
}

