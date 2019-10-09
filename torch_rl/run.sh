gpu=$1
shift

CUDA_VISIBLE_DEVICES="$gpu" nohup /data/anaconda/envs/py35/bin/python $@ &