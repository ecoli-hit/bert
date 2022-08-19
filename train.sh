#!/usr/bin/env bash
nvidia-smi


python3 -c "import torch; print(torch.__version__)"

src=de
tgt=en
bedropout=0.5
ARCH=transformer_s2_iwslt_de_en
DATAPATH=./iwslt_de_en
SAVEDIR=checkpoints/iwed_${src}_${tgt}_${bedropout}_300_real
mkdir -p $SAVEDIR
#if [ ! -f $SAVEDIR/checkpoint_nmt.pt ]
#then
#    cp /your_pretrained_nmt_model $SAVEDIR/checkpoint_nmt.pt
##fi
#if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
#then
#warmup="--warmup-from-nmt --reset-lr-scheduler --restore-file checkpoint_best.pt"
#else
warmup=""
#fi

export CUDA_VISIBLE_DEVICES=${1:-0}
python train.py $DATAPATH \
    -a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt  --label-smoothing 0.1 \
    --dropout 0.3 --max-tokens 4000 --fp16 --max-epoch 500 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --max-update 300000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9,0.98)' --save-dir $SAVEDIR --share-all-embeddings $warmup \
    --encoder-bert-dropout --encoder-bert-dropout-ratio $bedropout \
    --bert-model-name bert-base-german-dbmdz-uncased | tee -a $SAVEDIR/training.log