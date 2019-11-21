ROOT=../..
export PYTHONPATH=$PYTHONPATH:../..
mkdir snapshot
mkdir savecode
cd savecode
cp ../../../*.py .
cp -r ../../../subnet .
cd ..
CUDA_VISIBLE_DEVICES=0,1 python -u $ROOT/train.py --log log.txt --config config.json

