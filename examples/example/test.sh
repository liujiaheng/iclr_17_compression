ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir snapshot
CUDA_VISIBLE_DEVICES=7  python -u $ROOT/train.py --log log.txt --test --pretrain snapshot/iter2261420.model --config config.json
