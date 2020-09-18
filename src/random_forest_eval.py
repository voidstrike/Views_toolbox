import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def make_X_y(input_dataframe):
    y = input_dataframe['dv']
    X = input_dataframe.drop(columns=['d3mIndex', 'dv'])
    return X, y


def main(opt):
    train_path, test_path = opt.train, opt.test
    whole_train = pd.read_csv(train_path)
    whole_test = pd.read_csv(test_path)

    print('Preparing Dataset-------------------------------------')
    trainX, trainY = make_X_y(whole_train)
    testX, testY = make_X_y(whole_test)

    print('Training Model----------------------------------------')
    model = RandomForestRegressor(n_estimators=100, max_features='sqrt')
    model.fit(trainX, trainY)
    predY = model.predict(testX)

    print(mean_squared_error(predY, testY))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True, help='Path of the training path;')
    parser.add_argument('--test', type=str, required=True, help='Pathof the test path;')

    configuration = parser.parse_args()

    main(configuration)
