#!/bin/bash

MODELS=(
  # 'BRISQUE'
  # 'GMLOG'
  # 'HIGRADE1'
  # 'FRIQUEEALL'
  # 'CORNIA10K'
  # 'HOSA'
  # 'vgg19'
  # 'resnet50'
  # 'VBLIINDS_org'
  # 'TLVQM'
  # 'FFVIQE_release'
  'RAPIQUE'
)

DATASETS=(
  'LIVE_VQC'
  'KONVID_1K'
  'YOUTUBE_UGC'
)

for m in "${MODELS[@]}"
do
for DS in "${DATASETS[@]}"
do

  feature_file=mos_feat_files/${DS}_${m}_feats.mat
  mos_file=mos_feat_files/${DS}_metadata.csv
  out_file=result/${DS}_${m}_SVR_corr.mat
  log_file=logs/${DS}_regression.log

#   echo "$m" 
#   echo "${feature_file}"
#   echo "${out_file}"
#   echo "${log_file}"

  cmd="python evaluate_bvqa_features_regression.py"
  cmd+=" --model_name $m"
  cmd+=" --dataset_name ${DS}"
  cmd+=" --feature_file ${feature_file}"
  cmd+=" --mos_file ${mos_file}"
  cmd+=" --out_file ${out_file}"
  cmd+=" --log_file ${log_file}"
#   cmd+=" --use_parallel"
  cmd+=" --log_short"

  echo "${cmd}"

  eval ${cmd}
done
done
