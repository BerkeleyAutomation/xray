import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Make loss plots for model")
    parser.add_argument("csv_file", nargs="+", type=str, help="path to the csv file with losses")
    parser.add_argument("--output_path", type=str, help="path to store output images")
    args = parser.parse_args()

    output_path = os.path.dirname(args.csv_file[0]) if args.output_path is None else args.output_path
    f1, axes1 = plt.subplots(2, 2, figsize=(12.8, 9.6))
    f2, axes2 = plt.subplots(2, 2, figsize=(12.8, 9.6))

    for csvf in args.csv_file:
        data = pd.read_csv(csvf)

        # Make and save training plots
        sns.lineplot(x="epoch", y="train/loss", data=data, ax=axes1[0, 0])
        sns.lineplot(x="epoch", y="train/acc", data=data, ax=axes1[1, 0])
        sns.lineplot(x="epoch", y="train/bal_acc", data=data, ax=axes1[1, 1])
        sns.lineplot(x="epoch", y="train/mean_iou", data=data, ax=axes1[0, 1])

        # Make and save validation plots
        sns.lineplot(x="epoch", y="valid/loss", data=data[data["epoch"] > 1], ax=axes2[0, 0])
        sns.lineplot(x="epoch", y="valid/acc", data=data[data["epoch"] > 1], ax=axes2[1, 0])
        sns.lineplot(x="epoch", y="valid/bal_acc", data=data[data["epoch"] > 1], ax=axes2[1, 1])
        sns.lineplot(x="epoch", y="valid/mean_iou", data=data[data["epoch"] > 1], ax=axes2[0, 1])

    f1.savefig(os.path.join(output_path, "training_plots.png"), bbox_inches="tight")
    f2.savefig(os.path.join(output_path, "validation_plots.png"), bbox_inches="tight")
