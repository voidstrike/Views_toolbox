import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def make_X_y(input_dataframe):
    y = input_dataframe.filter(items=['dv', 'd3mIndex']).set_index('d3mIndex')
    # X = input_dataframe.drop(columns=['dv', 'in_africa']).set_index('d3mIndex')
    X = input_dataframe.drop(columns=['dv']).set_index('d3mIndex')
    return X, y


def main(opt):
    train_path, test_path, match_path = opt.train, opt.test, opt.match
    whole_train = pd.read_csv(train_path).dropna()
    whole_test = pd.read_csv(test_path)
    match_df = pd.read_csv(match_path)

    print('Preparing Dataset-------------------------------------')
    trainX, trainY = make_X_y(whole_train)
    testX, testY = make_X_y(whole_test)

    print('Training Model----------------------------------------')
    model = RandomForestRegressor(n_estimators=100, max_features='sqrt')
    # model = GradientBoostingRegressor(random_state=0)
    model.fit(trainX, trainY)

    predY = model.predict(testX)

    # Need another routine for Hurdle model

    if opt.model == 'dv':
        print('MSE: {}'.format(mean_squared_error(predY, testY)))
        predY = pd.DataFrame(data=predY).rename(columns={0: 'pred_dv'})
        predY.index = testY.index
        realY = testY

        out_df = pd.concat([realY, predY], axis=1).reset_index(drop=False)
        out_df = pd.merge(out_df, match_df, on=['d3mIndex'])
        # print(out_df.head(10))
        out_df = out_df.rename(columns={'date_t': 'month_id'})
        # out_df = out_df.set_index(['month_id', 'pg_id']).sort_index().drop(columns=['d3mIndex'])
        out_df = out_df.set_index(['month_id', 'country_id']).sort_index().drop(columns=['d3mIndex'])

        out_df.to_csv(opt.output)
    elif opt.model == 'hurdle':
        # original dv file is required
        dv_df = pd.read_csv('/home/yulin/Downloads/dv_data/dv_pgm_africa.csv').filter(items=['date_t', 'pg_id', 'dv'])
        dv_df = dv_df.set_index(['date_t', 'pg_id']).sort_index()
        dv_df = dv_df.loc[433:468]
        dv_df = dv_df.reset_index(drop=False)

        predY = pd.DataFrame(data=predY).rename(columns={0: 'pred_dv'})
        predY.index = testY.index

        predY = predY.reset_index(drop=False)
        predY = pd.merge(predY, match_df, on=['d3mIndex']).filter(items=['date_t', 'pg_id', 'pred_dv'])

        out_df = pd.merge(dv_df, predY, on=['date_t', 'pg_id'], how='left')
        # print(out_df.head(10))
        out_df = out_df.fillna(0)
        print('MSE: {}'.format(mean_squared_error(out_df.dv, out_df.pred_dv)))
        out_df = out_df.rename(columns={'date_t': 'month_id'})
        out_df = out_df.set_index(['month_id', 'pg_id']).sort_index()
        out_df.to_csv(opt.output)
    elif opt.model == 'cm':
        pass
    else:
        raise RuntimeError('Unsupported model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True, help='Path of the training file')
    parser.add_argument('--test', type=str, required=True, help='Path of the test file')
    parser.add_argument('--match', type=str, default='123', help='Path of the match.csv')
    parser.add_argument('--output', type=str, default='./exp_results.csv', help='Path of the output file')
    parser.add_argument('--model', type=str, default='dv', choices=['dv', 'hurdle'],
                        help='Model name, will affect the behavior of MSE computation and match method')

    configuration = parser.parse_args()

    main(configuration)
