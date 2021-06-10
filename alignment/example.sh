#!/bin/bash
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e
s=${1:-en}
t=${2:-es}
echo "Example based on the ${s}->${t} alignment"

if [ ! -d data/ ]; then
  mkdir -p data;
fi

if [ ! -d res/ ]; then
  mkdir -p res;
fi

dico_train=data/${s}-${t}.0-5000.txt
if [ ! -f "${dico_train}" ]; then
  DICO=$(basename -- "${dico_train}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi
dico_train1=data/${t}-${s}.0-5000.txt
if [ ! -f "${dico_train}" ]; then
  DICO=$(basename -- "${dico_train}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi

dico_test=data/${s}-${t}.5000-6500.txt
if [ ! -f "${dico_test}" ]; then
  DICO=$(basename -- "${dico_test}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi
dico_test1=data/${t}-${s}.5000-6500.txt
if [ ! -f "${dico_test}" ]; then
  DICO=$(basename -- "${dico_test}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi

src_emb=data/wiki.${s}.vec
if [ ! -f "${src_emb}" ]; then
  EMB=$(basename -- "${src_emb}")
  wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/${EMB}" -P data/
fi

tgt_emb=data/wiki.${t}.vec
if [ ! -f "${tgt_emb}" ]; then
  EMB=$(basename -- "${tgt_emb}")
  wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/${EMB}" -P data/
fi

output=res/wiki.${s}-${t}.vec

tune_dir=res/${s}-${t}
if [ ! -d "${tune_dir}" ]; then
  mkdir -p ${tune_dir}
fi

export CUDA_VISIBLE_DEVICES=1
# coefs=$(python3 -c "import numpy as np; print(' '.join(list(str(i) for i in np.linspace(1e-3,1e-1,20))))")
# echo -e "coef = $coefs"
# for c in ${coefs[@]}
# do
# echo -e "\ncoef = $c\n"
# python3 align_torch.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
#   --dico_train "${dico_train}" --dico_train1 "${dico_train1}"\
#   --dico_test "${dico_test}" --dico_test1 "${dico_test1}"\
#   --output "${output}" --tune_dir "${tune_dir}" \
#   --lr 25 \
#   --cycle_coef "${c}" --supervise intersect \
#   --batchsize 500 --niter 50 --nn_knn 1 --maxsup 500
# done
# python3 align.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
#   --dico_train "${dico_train}" \
#   --dico_test "${dico_test}" \
#   --output "${output}" \
#   --lr 25 --sgd 
python3 align_torch.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
  --dico_train "${dico_train}" --dico_train1 "${dico_train1}"\
  --dico_test "${dico_test}" --dico_test1 "${dico_test1}"\
  --output "${output}" --tune_dir "${tune_dir}" \
  --lr 25 \
  --cycle_coef 1e-3 --supervise intersect --mode baseline \
  --batchsize 500 --niter 50 --nn_knn 1 --maxsup 500
