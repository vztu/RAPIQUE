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
  'VBLIINDS'
  # 'TLVQM'
  # 'FFVIQE_release'
  'VIDEVAL'
  'RAPIQUE'
)

DATASETS=(
  'LIVE_VQA'
  # "LIVE_HFR"
)

for m in "${MODELS[@]}"
do
for DS in "${DATASETS[@]}"
do

  feature_file=feat_files/${DS}_${m}_feats.mat
  mos_file=mos_files/${DS}_metadata.csv
  out_file=result/${DS}_${m}_SVR_corr.mat
  log_file=logs/${DS}_regression.log

#   echo "$m" 
#   echo "${feature_file}"
#   echo "${out_file}"
#   echo "${log_file}"

  cmd="python evaluate_bvqa_features_by_content_regression.py"
  cmd+=" --model_name $m"
  cmd+=" --dataset_name ${DS}"
  cmd+=" --feature_file ${feature_file}"
  cmd+=" --mos_file ${mos_file}"
  cmd+=" --out_file ${out_file}"
  cmd+=" --log_file ${log_file}"
  cmd+=" --num_cont 10"  # 
  cmd+=" --num_dists 15"
#   cmd+=" --use_parallel"
  cmd+=" --log_short"
  cmd+=" --num_iterations 20"

  echo "${cmd}"

  eval ${cmd}
done
done
