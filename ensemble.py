import pandas as pd
import numpy as np

from utils import Ensemble
from utils.ensemble_class import Ensemble
import argparse

def main(args):
    file_list = sum(args.ENSEMBLE_FILES, [])
    
    if len(file_list) < 2:
        raise ValueError("Ensemble할 Model을 적어도 2개 이상 입력해 주세요.")
    
    en = Ensemble(filenames = file_list,filepath=args.RESULT_PATH)
    if args.ENSEMBLE_STRATEGY == 'SIMPLE':
        if args.ENSEMBLE_WEIGHT:
            weight_list = sum(args.ENSEMBLE_WEIGHT, [])
            strategy_title = 'sw-'+'-'.join(map(str,weight_list))
            output = en.simple_weighted(weight_list)
        else:
            raise ValueError("weight를 개수에 맞게 입력해주세요.")
    elif args.ENSEMBLE_STRATEGY == 'COMPLICATE':
        if args.ENSEMBLE_WEIGHT:
            ranked_weight=0.3
            weight_list = sum(args.ENSEMBLE_WEIGHT, [])
            strategy_title = 'cw-'+'-'.join(map(str,weight_list))
            output = en.complicated_weighted(weight_list,ranked_weight)
    elif args.ENSEMBLE_STRATEGY == 'RANK':
        if args.ENSEMBLE_WEIGHT:
            weight_list = sum(args.ENSEMBLE_WEIGHT, [])
            strategy_title = 'rw-'+'-'.join(map(str,weight_list))
            output = en.rank_weighted(weight_list)
        else:
            raise ValueError("weight를 개수에 맞게 입력해주세요.")
    else:
        pass
    files_title = '-'.join(file_list)
    output.to_csv(f'{args.RESULT_PATH}{files_title}-{strategy_title}.csv',index=False)
    print("complete saving ensemble file : "f'{files_title}-{strategy_title}.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    arg("--ENSEMBLE_FILES", nargs='+',required=True,
        type=lambda s: [item for item in s.split(',')],
        help='required: 앙상블할 submit 파일명을 쉼표(,)로 구분하여 모두 입력해 주세요. 이 때, .csv와 같은 확장자는 입력하지 않습니다.')
    arg('--ENSEMBLE_STRATEGY', type=str, default='SIMPLE',
        choices=['SIMPLE','COMPLICATE','RANK'],
        help='optional: [SIMPLE, COMPLICATE,RANK] 중 앙상블 전략을 선택해 주세요. (default="SIMPLE")')
    arg('--ENSEMBLE_WEIGHT', nargs='+',default=None,
        type=lambda s: [float(item) for item in s.split(',')],
        help='optional: Weighted 앙상블 전략에서 각 결과값의 가중치를 조정할 수 있습니다.')
    arg('--RESULT_PATH',type=str, default='../data/output/',
        help='optional: 앙상블할 파일이 존재하는 경로를 전달합니다. (default:"./data/output/")')
    args = parser.parse_args()
    main(args)