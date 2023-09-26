nohup python dnabert-2-117m-lstm-normal.py --gpu 0 --ssi > run0.log 2>&1 &
nohup python dnabert-2-117m-lstm-normal.py --gpu 1 --ssi --attention > run1.log 2>&1 &
nohup python dnabert-2-117m-lstm-normal.py --gpu 2 --ssi --lstm > run2.log 2>&1 &
nohup python dnabert-2-117m-lstm-normal.py --gpu 3 --ssi --lstm --attention > run3.log 2>&1 &

