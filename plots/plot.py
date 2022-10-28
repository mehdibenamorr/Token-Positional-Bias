import pandas as pd
import numpy as np
import matplotlib
import pandas as pd
import wandb

from dataset.ner_dataset import NERDataset, NERDatasetbuilder

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
from scipy.stats import norm

api = wandb.Api()

# sns.set(style='ticks', palette='Set2')
# colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
"""
1.5 columns (5,3.75) ; 1 column (3.54,2.65) ; 2 columns (7.25,5.43)
"""
style = {"figure.figsize": (5, 3.75),
         "figure.titlesize": 11,
         "legend.frameon": False,
         "legend.loc": 'upper right',
         "legend.fontsize": 11,
         "axes.labelsize": 11,
         "axes.titlesize": 11,
         "savefig.bbox": 'tight',
         "savefig.pad_inches": 0.05,
         "savefig.dpi": 300,
         "xtick.direction": 'in',
         "xtick.labelsize": 11,
         "xtick.major.size": 4,
         "xtick.major.width": 2,
         "xtick.minor.size": 2,
         "xtick.minor.width": 1,
         "xtick.minor.visible": True,
         "xtick.top": False,
         "xtick.bottom": True,
         "ytick.direction": 'in',
         "ytick.labelsize": 11,
         "ytick.major.size": 4,
         "ytick.major.width": 2,
         "ytick.minor.size": 2,
         "ytick.minor.width": 1,
         "ytick.minor.visible": True,
         "ytick.right": False,
         "ytick.left": True
         }
sns.set(context='paper', style='white', font_scale=1.5, color_codes=True, rc=style)

sns.set_theme(style="ticks")
plots_path = 'plots/files/'

def plot_pos_dist(data):
    f, ax = plt.subplots(figsize=(7.25, 5.43))
    # Plot the orbital period with horizontal boxes
    sns.boxplot(data=data, x="position", y="class",
                width=.4, palette="Set2", hue="f1",whis=[0, 100])
    # Tweak the visual presentation
    ax.set(ylabel="", xlabel='Positions')

    return f


def plot_loss_dist(data):
    f, ax = plt.subplots(figsize=(7.25, 5.43))
    # Plot the orbital period with horizontal boxes
    sns.boxplot(data=data, x="position", y="loss",
                width=.4, palette="Set2", whis=[0, 100])
    # Tweak the visual presentation
    ax.set(ylabel="Loss", xlabel='positions')

    return f
# Datasets
def dataset_plot():
    conll03 = NERDataset(dataset="conll03")
    ontonotes5 = NERDataset(dataset="ontonotes5")

    conll03_seq = pd.DataFrame(conll03.sequence_lengths, columns=["seq_lengths"])
    f, ax_hist = plt.subplots(1, figsize=(3.54, 2.65))
    sns.histplot(conll03_seq["seq_lengths"], ax=ax_hist,
                 edgecolor='black')

    # Class distribution
    def find_class_pos(labels, class_label, id2label):
        inds = []
        for i in range(len(labels)):
            if id2label[labels[i]] in [f"B-{class_label}", f"I-{class_label}", f"E-{class_label}", f"S-{class_label}"]:
                inds += i,
        return inds

    # CoNLL03
    train1 = conll03.train()
    classes = ["MISC", "ORG", "LOC", "PER"]
    id2label = conll03.id2label
    ner_tags = train1["ner_tags"]
    pos_dist = dict()
    for cls in classes:
        positions = []
        for labels in ner_tags:
            positions += find_class_pos(labels, cls, id2label)
        pos_dist[cls] = positions
    dfs = []
    for cls, pos in pos_dist.items():
        dfs += [(x, cls) for x in pos]
    dff = pd.DataFrame(dfs, columns=["position", "class"])

    f, ax = plt.subplots(figsize=(3.54,2.65))
    # Plot the orbital period with horizontal boxes
    sns.boxplot(data=dff, x="position", y="class",
                width=.4, palette="Set2", whis=[0, 100])
    # Tweak the visual presentation
    ax.set(ylabel="", xlabel='Positions')
    f.savefig(plots_path + 'conll03.pdf')


    #Ontonotes

    train2 = ontonotes5.train()
    classes = ['PERSON', 'GPE',
               'NORP',
               'CARDINAL', 'ORG',
               'DATE', 'LOC', 'EVENT', 'TIME',
               'PRODUCT', 'LANGUAGE', 'WORK_OF_ART', 'FAC', 'MONEY', 'QUANTITY', 'LAW', 'PERCENT']

    id2label = ontonotes5.id2label
    ner_tags = train2["ner_tags"]
    pos_dist2 = dict()
    for cls in classes:
        positions = []
        for labels in ner_tags:
            positions += find_class_pos(labels, cls, id2label)
        pos_dist2[cls] = positions
    dfs = []
    for cls, pos in pos_dist2.items():
        dfs += [(x, cls) for x in pos]
    dff = pd.DataFrame(dfs, columns=["position","class"])

    f, ax = plt.subplots(figsize=(7.25, 5.43))
    # Plot the orbital period with horizontal boxes
    sns.boxplot(data=dff, x="position", y="class",
                width=.4, palette="Set2", whis=[0, 100])
    # Tweak the visual presentation
    ax.set(ylabel="", xlabel='Positions')
    f.savefig(plots_path + 'ontonotes.pdf')


def bias_experiment(experiment="bert_position_bias_synthetic", dataset="ontonotes5"):
    labels = NERDatasetbuilder.get_labels(dataset=dataset)
    labels = list(np.unique([l.split("-")[-1] for l in labels]))
    entity = "benamor"  # set to your entity and project
    runs = api.runs(entity + "/" + experiment + "-" + dataset)


    # Per Class distribution f1 score for max and min
    # api.runs(
    #     path="my_entity/my_project",
    #     filters={"display_name": {"$regex": "^foo.*"}}
    # )

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        if run.state == "finished":
            summary_list.append(run.summary._json_dict)
            df = run.history()
            dfs = []
            for k in range(1,11):
                df_ = run.history(keys=[ key for key in  df.keys() if key.startswith(f"test/k={k}_")])
                df_.columns = [col.split(f"={k}_")[-1] for col in df_.columns]
                df_["k"] = k
                dfs.append(df_)

            results = pd.concat(dfs)

            f, ax = plt.subplots(figsize=(3.54,2.65))
            # Plot the orbital period with horizontal boxes
            sns.lineplot(data=results, x="k", y="overall_f1",
                         palette="Set2", markers=True)
            # Tweak the visual presentation
            ax.set(ylabel="F1", xlabel='k-factor')
            f.savefig(plots_path + f'{dataset}_f1.jpg')
            for cls in labels:
                if cls != "O":
                    f, ax = plt.subplots(figsize=(3.54,2.65))
                    # Plot the orbital period with horizontal boxes
                    sns.lineplot(data=results, x="k", y=f"{cls}.f1",
                                 palette="Set2", markers=True)
                    # Tweak the visual presentation
                    ax.set(ylabel=f"F1({cls})", xlabel='k-factor')
                    f.savefig(plots_path + f'{dataset}_{cls}_f1.jpg')


if __name__ == "__main__":
    # dataset_plot()
    bias_experiment()
