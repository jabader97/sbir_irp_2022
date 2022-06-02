#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --time=3-00:00
#SBATCH -o /mnt/qb/akata/jbader40/logs/test_job_%j.out
#SBATCH -e /mnt/qb/akata/jbader40/logs/test_job_%j.err
#SBATCH -x slurm-bm-54,slurm-bm-20,slurm-bm-28,slurm-bm-37,slurm-bm-63,slurm-bm-22,slurm-bm-83

python src/train.py --seed 0 --batch-size 32 --group sem_pcyc --model sem_pcyc --dataset TU-Berlin --dim-out 64 --semantic-models word2vec-google-news --epochs 200 --early-stop 200 --lr 0.0001 --path_aux /mnt/qb/akata/jbader40/sbir/sem_pcyc/pretrained_models --path_dataset /mnt/qb/akata/jbader40/sbir/sem_pcyc/datasets --log_online --wandb_key 1cdc17e811df70a17e4d9174c95f5b4e9f4a01dc --project sbir



