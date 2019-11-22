import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import seaborn as sns
sns.set()
import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Make loss plots for model")
    parser.add_argument('csv_file', type=str, help="path to the csv file with losses")
    parser.add_argument('--output_path', type=str, help="path to store output images")
    parser.add_argument('--sample', type=int, default=10, help='downsample rate for plots')
    args = parser.parse_args()

    output_path = os.path.dirname(args.csv_file) if args.output_path is None else args.output_path
    data = pd.read_csv(args.csv_file)
    subsampled_data = data[data['iteration'] % args.sample == 0]

    # Make and save training plots
    f, axes = plt.subplots(2, 2, figsize=(12.8, 9.6))
    # for ax in axes.flatten()[1:]:
    #     ax.set_ylim((0,1))
    sns.lineplot(x='iteration', y='train/loss', data=subsampled_data, ax=axes[0,0])
    sns.lineplot(x='iteration', y='train/acc', data=subsampled_data, ax=axes[1,0])
    sns.lineplot(x='iteration', y='train/bal_acc', data=subsampled_data, ax=axes[1,1])
    sns.lineplot(x='iteration', y='train/mean_iou', data=subsampled_data, ax=axes[0,1])
    plt.savefig(os.path.join(output_path, 'training_plots.png'), bbox_inches='tight')

    # Make and save validation plots
    f, axes = plt.subplots(2, 2, figsize=(12.8, 9.6))
    # for ax in axes.flatten()[1:]:
    #     ax.set_ylim((0,1))
    sns.lineplot(x='epoch', y='valid/loss', data=data, ax=axes[0,0])
    sns.lineplot(x='epoch', y='valid/acc', data=data, ax=axes[1,0])
    sns.lineplot(x='epoch', y='valid/bal_acc', data=data, ax=axes[1,1])
    sns.lineplot(x='epoch', y='valid/mean_iou', data=data, ax=axes[0,1])
    plt.savefig(os.path.join(output_path, 'validation_plots.png'), bbox_inches='tight')
