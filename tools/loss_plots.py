import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import argparse
import seaborn as sns
sns.set()
import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Make loss plots for model")
    parser.add_argument('csv_file', nargs='+', type=str, help="path to the csv file with losses")
    parser.add_argument('--output_path', type=str, help="path to store output images")
    parser.add_argument('--sample', type=int, default=10, help='downsample rate for plots')
    args = parser.parse_args()

    output_path = os.path.dirname(args.csv_file[0]) if args.output_path is None else args.output_path
    f1, axes1 = plt.subplots(2, 2, figsize=(12.8, 9.6))
    f2, axes2 = plt.subplots(2, 2, figsize=(12.8, 9.6))

    for csvf in args.csv_file:
        data = pd.read_csv(csvf)
        # data = data[np.logical_and(data['iteration'] % args.sample == 0, data['iteration'] > 10)]

        # Make and save training plots
        # for ax in axes.flatten()[1:]:
        #     ax.set_ylim((0,1))
        sns.lineplot(x='epoch', y='train/loss', data=data, ax=axes1[0,0])
        sns.lineplot(x='epoch', y='train/acc', data=data, ax=axes1[1,0])
        sns.lineplot(x='epoch', y='train/bal_acc', data=data, ax=axes1[1,1])
        sns.lineplot(x='epoch', y='train/mean_iou', data=data, ax=axes1[0,1])

        # Make and save validation plots
        # for ax in axes.flatten()[1:]:
        #     ax.set_ylim((0,1))
        sns.lineplot(x='epoch', y='valid/loss', data=data[data['epoch'] > 1], ax=axes2[0,0])
        sns.lineplot(x='epoch', y='valid/acc', data=data[data['epoch'] > 1], ax=axes2[1,0])
        sns.lineplot(x='epoch', y='valid/bal_acc', data=data[data['epoch'] > 1], ax=axes2[1,1])
        sns.lineplot(x='epoch', y='valid/mean_iou', data=data[data['epoch'] > 1], ax=axes2[0,1])
    
    # custom_lines = [Line2D([0], [0], color=sns.color_palette()[0], lw=4),
    #                 Line2D([0], [0], color=sns.color_palette()[1], lw=4),
    #                 Line2D([0], [0], color=sns.color_palette()[2], lw=4),
    #                 Line2D([0], [0], color=sns.color_palette()[3], lw=4)]
    # f2.legend(custom_lines, ['Aspect Ratio 1', 'Aspect Ratio 2', 'Aspect Ratio 5', 'Aspect Ratio 10'], 
    #           bbox_to_anchor=(0.2,0.82,0.6,0.2), loc='center', mode="expand", borderaxespad=0, ncol=4)
    # f1.legend(custom_lines, ['Aspect Ratio 1', 'Aspect Ratio 2', 'Aspect Ratio 5', 'Aspect Ratio 10'], 
    #           bbox_to_anchor=(0.2,0.82,0.6,0.2), loc='center', mode="expand", borderaxespad=0, ncol=4)
    f1.savefig(os.path.join(output_path, 'training_plots.png'), bbox_inches='tight')
    f2.savefig(os.path.join(output_path, 'validation_plots.png'), bbox_inches='tight')
    # plt.show()
