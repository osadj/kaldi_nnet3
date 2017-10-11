#!/bin/bash

# Author: Omid Sadjadi <omid.sadjadi@ieee.org>
# Based on Kaldi scripts by D. Povey, P. Ghahremani and others

# NOTE1: Kaldi should be installed before running this script
# NOTE2: You should first run the Kaldi recipe for e.g., fisher_swbd to
#        generate alignments for swbd (either tri4a or tri5a is ok)
# This script can perform the following operations:
# 1. makes ark feature files from htk feature files (also generates CMVN stats)
# 2. generate egs for nnet3
# 3. generate a config file for nnet3 model training
# 4. train a raw Neural Network with a BN layer
# 5. remove the extra layers after the BN layer

echo "$0 $@"  # Print the command line for logging
. ./cmd.sh
. ./path.sh
set -e

make_ark=false
lay_egs=true
make_config=true
train_dnn=true
remove_egs=false
rm_extra_layers=false
cmd=run.pl
srand=0
stage=0
train_stage=-10
get_egs_stage=0
feat_type=raw
cmvn_opts=
online_ivector_dir=
splice_width=10
samples_per_iter=400000
egs_opts=
egs_dir=

data=$1 #data/train_swbd/
lang=$2 #data/lang
alidir=$3 #exp/tri4a_swb_ali/
dir=$4 #exp/nnet3/swbd/

if [ "$make_ark" = true ]; then
  features_dir=my_features # this is where you stored your features in HTK format
  ./make_ark.sh $data $features_dir/
fi

extra_opts=()
#[ ! -z "$cmvn_opts" ] && extra_opts+=(--cmvn-opts "$cmvn_opts")
#[ ! -z "$feat_type" ] && extra_opts+=(--feat-type $feat_type)
[ ! -z "$online_ivector_dir" ] && extra_opts+=(--online-ivector-dir $online_ivector_dir)
extra_opts+=(--cmvn-opts "--norm-means=true --norm-vars=true")
extra_opts+=(--left-context $splice_width)
extra_opts+=(--right-context $splice_width)

if [ -z $egs_dir ]; then
  egs_dir=$dir/egs
fi

if [ "$lay_egs" = true ]; then
  echo "$0: calling get_egs.sh"
  steps/nnet3/get_egs.sh $egs_opts "${extra_opts[@]}" \
      --samples-per-iter $samples_per_iter \
      --stage $get_egs_stage \
      --cmd "$cmd" $egs_opts \
      --nj 20 \
      $data $alidir $egs_dir || exit 1;
fi

num_leaves=`am-info $alidir/final.mdl 2>/dev/null | awk '/number of pdfs/{print $NF}'` || exit 1;
feat_dim=`feat-to-dim scp:$data/feats.scp -`

if [ "$make_config" = true ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  echo "input feature dimension: $feat_dim"
  echo "number of nodes in the output layer: $num_leaves"

  hidden_layer_dim=1024

  if [ -z $bnf_dim ]; then
    bnf_dim=80
  fi
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=$feat_dim name=input
  relu-renorm-layer name=dnn1 input=Append(-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10) dim=$hidden_layer_dim
  relu-renorm-layer name=dnn2 dim=$hidden_layer_dim
  relu-renorm-layer name=dnn3 dim=$hidden_layer_dim
  relu-renorm-layer name=dnn4 dim=$hidden_layer_dim
  relu-renorm-layer name=dnn5 dim=$hidden_layer_dim
  renorm-layer name=dnn_bn dim=$bnf_dim
  relu-renorm-layer name=prefinal-affine input=dnn_bn dim=1024
  output-layer name=output dim=$num_leaves max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/
fi

if [ "$train_dnn" = true ]; then
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=true --norm-vars=true" \
    --trainer.num-epochs 12 \
    --trainer.optimization.num-jobs-initial=6 \
    --trainer.optimization.num-jobs-final=12 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=512,256 \
    --trainer.samples-per-iter=400000 \
    --trainer.max-param-change=2.0 \
    --trainer.srand=$srand \
    --feat-dir $data \
    --targets-scp $alidir \
    --nj 20 \
    --egs.dir $egs_dir \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 50 \
    --use-dense-targets false \
    --use-gpu true \
    --dir=$dir  || exit 1;

fi


if [ "$rm_extra_layers" = true ]; then
  nnet3-copy --binary=false "--nnet-config=echo output-node name=output input=dnn_bn.renorm |" \
             --edits=remove-orphans $dir/final.raw $dir/final.txt
fi
