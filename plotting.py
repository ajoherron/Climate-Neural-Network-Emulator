import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs

from tools import t_test


def plot_training_loss(EPOCHS, all_train_losses):
    plt.plot(range(EPOCHS), all_train_losses[0], label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.show()


def global_timeseries_plot(Y_pred, Y_train, Y_test, VARIABLE):

    # Gather weights
    weights = np.cos(np.deg2rad(Y_pred.lat))
    weights.name = "weights"
    weights.mean()

    # Initialize a flag to track if the label has been added
    label_added = False

    # Adds traininng data plots
    for i in range(len(Y_train)):
        if not label_added:
            Y_train[i][VARIABLE[0]].weighted(weights).mean(dim=["lat", "lon"]).plot(
                color="grey", label="Training Data"
            )
            label_added = True  # Set the flag to True after adding the label
        else:
            Y_train[i][VARIABLE[0]].weighted(weights).mean(dim=["lat", "lon"]).plot(
                color="grey"
            )

    # Plot model predictions
    Y_pred[VARIABLE[0]].weighted(weights).mean(dim=["lat", "lon"]).plot(
        color="black", label="Model Predictions"
    )

    # Plot test output
    Y_test[VARIABLE[0]].weighted(weights).mean(dim=["lat", "lon"]).plot(
        color="green", label="Test Data (SSP245)"
    )

    # Formatting
    plt.title("Global Average Timeseries")
    plt.xlabel("Year")
    plt.ylabel("Precipitation (mm/day)")
    plt.legend()
    plt.grid()


def global_anomaly_plot(Y_pred, Y_test, p_value, VARIABLE):

    # Extract data for the average of 2080-2100
    average_of_runs = Y_pred
    prediction_tsurf = average_of_runs
    prediction_tsurf_2080_2100 = prediction_tsurf.sel(year=slice(2080, 2100)).mean(
        dim="year"
    )[VARIABLE[0]]
    validation_tsurf_2080_2100 = (
        Y_test[VARIABLE[0]].sel(year=slice(2080, 2100)).mean(dim="year")
    )
    anomaly = prediction_tsurf_2080_2100 - validation_tsurf_2080_2100

    # Perform significance test (copied from WP)
    diff = Y_pred.sel(year=slice(2080, 2100))[VARIABLE[0]]
    diff_mean = diff.mean(dim=["year"]).values
    diff_std = diff.std(dim=["year"]).values
    diff_num = diff.count(dim=["year"]).values

    # Perform t-test
    _, p = t_test(diff_mean, diff_std, diff_num)

    # Plot anomaly
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, projection=ccrs.Robinson())
    anomaly.where(p < p_value).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        cbar_kwargs={"label": "mm/day"},
    )

    # Figure out how to include this
    # Add coastlines
    # ax.coastlines()

    plt.title("Global Emulator Anomalies")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
