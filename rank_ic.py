import pandas as pd
import argparse

def rank_ic(result_path, label_path):

    test_label = pd.read_csv(label_path).set_index(['time_id', 'stock_id'])
    pred = pd.read_csv(result_path).set_index(['time_id', 'stock_id'])
    result = pd.concat([pred, test_label], axis=1)

    rank_ic = result.groupby('time_id').apply(lambda df: (df['pred'].rank()).corr(df['label'].rank())).mean()
    return rank_ic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str,  default='../test_label.csv')
    parser.add_argument('--result_path', type=str,  default='./result.csv')
    args = parser.parse_args()

    rank_ic = rank_ic(args.result_path, args.label_path)
    print('rank_ic: ', rank_ic)


