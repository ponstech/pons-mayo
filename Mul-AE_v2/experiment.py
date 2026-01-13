import os
import argparse
import subprocess
from datetime import datetime
from unicodedata import name

parser = argparse.ArgumentParser(description='Finetune Training')

parser.add_argument('--name', required=True, type=str, choices=['BP4D', 'DISFA'])
# parser.add_argument('--ratio', required=True, type=str, choices=['0.0','0.25', '0.5', '0.75'])
# parser.add_argument('--epoch', required=True, type=str, choices=['all', '60', '50', '40', '30', '20', '10'])



def main():
    gpu = 0
    args = parser.parse_args()
    now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

    for fold in ['fold1','fold2','fold3']:
        log=f"hydra.run.dir=/media/xiang/Disk3/Work/logs/Mul-AE/Experiment/{args.name}/{now}/{fold}"

        print(f'==========START============== {fold} ==========START==============')
        cmd = f'python finetune.py save_epochs=3 gpu_ids={gpu}  fold={fold} {log}'
        subprocess.call(cmd, shell=True)
        print(f'==========END================ {fold} ============END==============')

    return


    if args.name == 'DISFA':
        now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        for fold in ['fold1','fold2','fold3']:
            log=f"hydra.run.dir=D:/Work/Xiang/logs/Mul-AE/Experiment/{args.name}/{now}/{fold}"

            print(f'==========START============== {fold} ==========START==============')
            cmd = f'python finetune.py save_epochs=3 gpu_ids={gpu}  fold={fold} {log}'
            subprocess.call(cmd, shell=True)
            print(f'==========END================ {fold} ============END==============')

    epoch_list = []
    if args.epoch == 'all':
        epoch_list = ['30', '25', '20', '15', '10', '5']
    else:
        epoch_list = [args.epoch]

    

    if args.name == 'BP4D':    
        for epoch in epoch_list:
            finetune= f"D:/Work/Xiang/ACMMM_2022/saved_model/pretrain/new/BP4D/all_model_{args.ratio}/checkpoint-epoch{epoch}.pth"
            now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
            for fold in ['fold1','fold2','fold3']:
                log=f"hydra.run.dir=D:/Work/Xiang/logs/Mul-AE/Experiment/{args.name}/bp4d+_mask_{args.ratio}/{now}/{fold}"
 
                print(f'==========START============== {fold} ==========START==============')
                cmd = f'python finetune.py save_epochs=3 gpu_ids={gpu} arch.finetune={finetune}  fold={fold} {log}'
                subprocess.call(cmd, shell=True)
                print(f'==========END================ {fold} ============END==============')


        

if __name__ == '__main__':
    main()