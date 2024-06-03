import numpy as np


def normalize_data(data, var, meanstd_dict):
    mean = meanstd_dict[var][0]
    std = meanstd_dict[var][1]
    return (data - mean) / std


def calculate_mean_std(X_train, INPUT_LIST):
    meanstd_inputs = {}
    for var in INPUT_LIST:
        array = np.concatenate([X_train[i][var].data for i in range(len(X_train))])
        meanstd_inputs[var] = (array.mean(), array.std())
    return meanstd_inputs


def apply_normalization(X_train, X_test, meanstd_inputs, INPUT_LIST):
    X_train_norm = []
    for i, train_xr in enumerate(X_train):
        for var in INPUT_LIST:
            var_dims = train_xr[var].dims
            train_xr = train_xr[INPUT_LIST].assign(
                {
                    var: (
                        var_dims,
                        normalize_data(train_xr[var].data, var, meanstd_inputs),
                    )
                }
            )
        X_train_norm.append(train_xr)
    X_test_xr = X_test[0][INPUT_LIST]
    return X_train_norm, X_test_xr


def reshape_training_input(X_train_xr, slider=0):
    X_train_np = X_train_xr.to_array().transpose("year", "lat", "lon", "variable").data
    time_length = X_train_np.shape[0]
    X_train_to_return = np.array(
        [X_train_np[i : i + slider] for i in range(0, time_length - slider + 1)]
    )
    return X_train_to_return


def reshape_training_output(Y_train_xr, var, slider=0):
    Y_train_np = Y_train_xr[var[0]].data
    time_length = Y_train_np.shape[0]
    Y_train_to_return = np.array(
        [[Y_train_np[i + slider - 1]] for i in range(0, time_length - slider + 1)]
    )
    return Y_train_to_return


def merge_training_data(X_train, Y_train, X_train_norm, VARIABLE, SLIDER_LENGTH):

    # Concatenate input data
    X_train_all = np.concatenate(
        [
            reshape_training_input(X_train_norm[i], slider=SLIDER_LENGTH)
            for i in range(len(X_train))
        ],
        axis=0,
    )
    X_train_all = X_train_all[:, 0, :, :, :]  # Select the first channel

    # Concatenate target data
    Y_train_all = np.concatenate(
        [
            reshape_training_output(Y_train[i], VARIABLE, slider=SLIDER_LENGTH)
            for i in range(len(X_train))
        ],
        axis=0,
    )
    Y_train_all = Y_train_all[:, 0, :, :]  # Select the first channel
    return X_train_all, Y_train_all
