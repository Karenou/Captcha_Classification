#!/bin/bash
# pip install cleverhans

python3 classification.py

python3 attack.py --eps=0.1 --sim_eps=0.1 --sparse_eps=20 --pgd_nb_iter=50 --model=cnn
python3 attack.py --eps=0.2 --sim_eps=0.2 --sparse_eps=30 --pgd_nb_iter=50 --model=cnn

python3 attack.py --eps=0.1 --sim_eps=0.1 --sparse_eps=20 --pgd_nb_iter=50 --model=res18
python3 attack.py --eps=0.2 --sim_eps=0.2 --sparse_eps=30 --pgd_nb_iter=50 --model=res18

python3 attack.py --eps=0.1 --sim_eps=0.1 --sparse_eps=20 --pgd_nb_iter=50 --model=res50
python3 attack.py --eps=0.2 --sim_eps=0.2 --sparse_eps=30 --pgd_nb_iter=50 --model=res50

python3 attack.py --eps=0.1 --sim_eps=0.1 --sparse_eps=20 --pgd_nb_iter=50 --model=mobile
python3 attack.py --eps=0.2 --sim_eps=0.2 --sparse_eps=30 --pgd_nb_iter=50 --model=mobile

python3 attack.py --eps=0.1 --sim_eps=0.1 --sparse_eps=20 --pgd_nb_iter=50 --model=dense161
python3 attack.py --eps=0.2 --sim_eps=0.2 --sparse_eps=30 --pgd_nb_iter=50 --model=dense161

python3 evaluate.py