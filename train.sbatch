timestamp=$(date +%Y%m%d-%H-%M)
echo $timestamp

python train.py $timestamp
python eval.py $timestamp
python visualize.py $timestamp
