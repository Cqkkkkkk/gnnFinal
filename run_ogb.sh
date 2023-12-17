# python main_ogb.py --stages 300 300 300 300 --num-hops 2 --label-feats --num-label-hops 2 \
# 	--n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
# 	--lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --amp --seeds 1

# python main_ogb.py --cfg configs/ogb-mag.yaml
python main_ogb.py --cfg configs/mag-year.yaml