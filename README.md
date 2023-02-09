# STGN Experiment on NoisyNER

Sub-experiment of ["STGN: an Implicit Regularization Method for Learning with Noisy Labels in Natural Language Processing"](https://aclanthology.org/2022.emnlp-main.515/) (EMNLP 2022) by Tingting Wu, Xiao Ding, Minji Tang, Hao Zhang, Bing Qin, Ting Liu.

Main experiment: [github](https://github.com/tangminji/STGN-sst)

## Data: NoisyNER
Data Source: ["Analysing the Noise Model Error for Realistic Noisy Label Data"](https://github.com/uds-lsv/noise-estimation) (AAAI 2021) by Hedderich, Zhu & Klakow

This experiment was adapt from "https://github.com/uds-lsv/noise-estimation/tree/master/base_model_performance/exp_ner", and you should follow the instruction of original README.

## Run Code
We give an example shell:
```
i=1
method=base

for j in 0 1 2
do
    python tm_train_hy_nruns.py \
    --dataset NoisyNER \
    --exp_name ../nrun/ner_$method/$i/seed$j/ \
    --params_path best_params$i.json \
    --out_tmp ner_out_tmp$i.json \
    --sub_script sbatch_ner_hy_sub$i.sh \
    --output_dir ../logger/ \
    --label_set $i \
    --seed $j
done
```

Options:
+ label_set(i): 1,2,3,4,5,6,7
+ method: base, STGN

You should run the tm_train_hy_nruns.py with proper params written in json files.