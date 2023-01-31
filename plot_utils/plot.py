# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import os
import torch

from matplotlib.offsetbox import AnchoredText

from dataset.ner_dataset import NERDataset, NERDatasetbuilder
from dataset.pos_dataset import POSDataset, POSDatasetbuilder

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
plots_dir = '/home/mehdi/Desktop/Workspace/plots/'
os.makedirs(plots_dir, exist_ok=True)


def plot_pos_dist(data):
    f, ax = plt.subplots(figsize=(7.25, 5.43))
    # Plot the orbital period with horizontal boxes
    sns.boxplot(data=data, x="position", y="class",
                width=.4, palette="Set2", hue="f1", whis=[0, 100], ax=ax)
    # Tweak the visual presentation
    ax.set(ylabel="", xlabel='Positions')

    return f


def plot_loss_dist(data):
    f, ax = plt.subplots(figsize=(7.25, 5.43))
    # Plot the orbital period with horizontal boxes
    sns.kdeplot(data=data, x="position", y="loss", kind="kde", ax=ax)
    # Tweak the visual presentation
    ax.set(ylabel="Loss", xlabel='positions')

    return f


# Datasets
def dataset_plot():
    save_dir = os.path.join(plots_dir, "datasets")
    os.makedirs(save_dir, exist_ok=True)
    conll03 = NERDataset(dataset="conll03")
    ontonotes5 = NERDataset(dataset="ontonotes5")
    en_ewt = POSDataset(dataset="en_ewt")
    tweebank = POSDataset(dataset="tweebank")

    # Datasets statistics
    conll03_seq = pd.DataFrame(conll03.sequence_lengths, columns=["seq_lengths"])
    f, ax_hist = plt.subplots(1, figsize=(3.54, 2.65))
    sns.histplot(conll03_seq["seq_lengths"], ax=ax_hist,
                 edgecolor='black')

    ax_hist.set(xlabel='Sequence lengths', ylabel='Count')

    plt.legend()
    f.savefig(save_dir + '/conll03_seq_lengths.pdf')

    ontonotes5_seq = pd.DataFrame(ontonotes5.sequence_lengths, columns=["seq_lengths"])
    f, ax_hist = plt.subplots(1, figsize=(3.54, 2.65))
    sns.histplot(ontonotes5_seq["seq_lengths"], ax=ax_hist,
                 edgecolor='black', bins=50)

    ax_hist.set(xlabel='Sequence lengths', ylabel='Count')

    plt.legend()
    f.savefig(save_dir + '/ontonotes5_seq_lengths.pdf')

    en_ewt_seq = pd.DataFrame(en_ewt.sequence_lengths, columns=["seq_lengths"])
    f, ax_hist = plt.subplots(1, figsize=(3.54, 2.65))
    sns.histplot(en_ewt_seq["seq_lengths"], ax=ax_hist,
                 edgecolor='black')

    ax_hist.set(xlabel='Sequence lengths', ylabel='Count')

    plt.legend()
    f.savefig(save_dir + '/en_ewt_seq_lengths.pdf')

    tweebank_seq = pd.DataFrame(tweebank.sequence_lengths, columns=["seq_lengths"])
    f, ax_hist = plt.subplots(1, figsize=(3.54, 2.65))
    sns.histplot(tweebank_seq["seq_lengths"], ax=ax_hist,
                 edgecolor='black')

    ax_hist.set(xlabel='Sequence lengths', ylabel='Count')

    plt.legend()
    f.savefig(save_dir + '/tweebank_seq_lengths.pdf')

    df = pd.DataFrame(
        [(x, "conll03") for x in conll03.sequence_lengths] + [(x, "ontonotes5") for x in
                                                              ontonotes5.sequence_lengths] + [(x, "en_ewt") for x in
                                                                                              en_ewt.sequence_lengths] + [
            (x, "tweebank") for x in tweebank.sequence_lengths],
        columns=["seq_lengths", "dataset"])

    benchmarks = {"conll03": "CoNLL03", "ontonotes5": "OntoNotes5.0", "en_ewt": "UD_en", "tweebank": "TweeBank"}
    df["dataset"] = df["dataset"].map(benchmarks)
    # Histogram
    f, ax = plt.subplots(figsize=(5, 3.75))
    sns.histplot(data=df[df["seq_lengths"] < 100], x="seq_lengths", hue="dataset", palette="Set2", multiple="dodge",
                 bins=20)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels_ = [str(int(int(l) / 1000)) for l in labels]
    ax.set_yticklabels(labels_)
    ax.set(xlabel='Word Positions', ylabel='Count ($10^3$)')
    f.savefig(save_dir + '/hist_seq_lengths.pdf')

    f, ax = plt.subplots(figsize=(5.25, 2.43))

    # Plot the orbital period with horizontal boxes
    sns.boxplot(x="seq_lengths", y="dataset", data=df,
                width=.3, palette="vlag", whis=[0, 100])

    # Tweak the visual presentation
    ax.set(ylabel="", xlabel='Sequence length')
    f.savefig(save_dir + '/seq_lengths.pdf')

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
    conll03_cls = pd.DataFrame(dfs, columns=["position", "NE tag"])

    # Histogram
    f, ax = plt.subplots(figsize=(3.54, 2.65))
    # Plot the orbital period with horizontal boxes
    sns.histplot(data=conll03_cls.loc[conll03_cls["NE tag"].isin(["PER", "MISC"])], x="position", hue="NE tag",
                 palette="Set2",
                 multiple="layer", stat="count", bins=20)
    # Tweak the visual presentation
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels_ = [str(int(int(l) / 1000)) for l in labels]
    ax.set_yticklabels(labels_)
    ax.set(ylabel="Count ($10^3$)", xlabel='Positions')
    f.savefig(save_dir + '/conll03_class_dist.pdf')

    f, ax = plt.subplots(figsize=(3.54, 2.65))
    # Plot the orbital period with horizontal boxes
    sns.boxplot(data=conll03_cls, x="position", y="NE tag",
                width=.4, palette="colorblind", whis=[0, 100])
    # Tweak the visual presentation
    ax.set(ylabel="", xlabel='Positions')
    f.savefig(save_dir + '/conll03.pdf')

    # Ontonotes

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
    ontonotes_cls = pd.DataFrame(dfs, columns=["position", "NE tag"])

    # Histogram
    f, ax = plt.subplots(figsize=(3.54, 2.65))
    # Plot the orbital period with horizontal boxes
    sns.histplot(data=ontonotes_cls.loc[ontonotes_cls["NE tag"].isin(["LAW", "EVENT", "WORK_OF_ART"])], x="position",
                 hue="NE tag",
                 palette="Set2",
                 multiple="layer", stat="count", bins=20, hue_order=["LAW", "EVENT", "WORK_OF_ART"])
    ax.set(ylabel="Count", xlabel='Positions')
    f.savefig(save_dir + '/ontonotes_class_dist.pdf')

    f, ax = plt.subplots(figsize=(7.25, 5.43))
    # Plot the orbital period with horizontal boxes
    sns.boxplot(data=ontonotes_cls, x="position", y="NE tag",
                width=.4, palette="colorblind", whis=[0, 100])
    # Tweak the visual presentation
    ax.set(ylabel="", xlabel='Positions')
    f.savefig(save_dir + '/ontonotes.pdf')

    # TWEETBANK
    train3 = tweebank.train()
    classes = ["ADJ", "NOUN", "VERB", "ADV", "PRON", "DET", "ADP", "NUM", "CCONJ", "X", "INTJ", "SYM", "PUNCT", "PART",
               "AUX", "PROPN", "SCONJ"]
    id2label = tweebank.id2label
    pos_tags = train3["pos_tags"]
    pos_dist3 = dict()
    for cls in classes:
        positions = []
        for labels in pos_tags:
            positions += find_class_pos(labels, cls, id2label)
        pos_dist3[cls] = positions
    dfs = []
    for cls, pos in pos_dist3.items():
        dfs += [(x, cls) for x in pos]
    tweetbank_cls = pd.DataFrame(dfs, columns=["position", "POS tag"])

    # Histogram
    f, ax = plt.subplots(figsize=(3.54, 2.65))
    # Plot the orbital period with horizontal boxes
    sns.histplot(data=tweetbank_cls.loc[tweetbank_cls["POS tag"].isin(["PROPN", "NOUN", "ADJ"])], x="position",
                 hue="POS tag",
                 palette="Set2",
                 multiple="layer", stat="count", bins=20, hue_order=["ADJ", "PROPN", "NOUN"])
    ax.set(ylabel="Count", xlabel='Positions')
    f.savefig(save_dir + '/tweetbank_class_dist.pdf')

    f, ax = plt.subplots(figsize=(7.25, 5.43))
    # Plot the orbital period with horizontal boxes
    sns.boxplot(data=tweetbank_cls, x="position", y="POS tag",
                width=.4, palette="Set2", whis=[0, 100])
    # Tweak the visual presentation
    ax.set(ylabel="", xlabel='Positions')
    f.savefig(save_dir + '/tweetbank.pdf')

    # UD
    train4 = en_ewt.train()
    classes = ["ADJ", "NOUN", "VERB", "ADV", "PRON", "DET", "ADP", "NUM", "CCONJ", "X", "INTJ", "SYM", "PUNCT", "PART",
               "AUX", "PROPN", "SCONJ"]
    id2label = en_ewt.id2label
    pos_tags = train4["pos_tags"]
    pos_dist4 = dict()
    for cls in classes:
        positions = []
        for labels in pos_tags:
            positions += find_class_pos(labels, cls, id2label)
        pos_dist4[cls] = positions
    dfs = []
    for cls, pos in pos_dist4.items():
        dfs += [(x, cls) for x in pos]
    ud_cls = pd.DataFrame(dfs, columns=["position", "POS tag"])

    # Histogram
    f, ax = plt.subplots(figsize=(3.54, 2.65))
    # Plot the orbital period with horizontal boxes
    sns.histplot(data=ud_cls.loc[ud_cls["POS tag"].isin(["PRON", "NOUN"])], x="position",
                 hue="POS tag",
                 palette="Set2",
                 multiple="layer", stat="count", bins=20, hue_order=["PRON", "NOUN"])
    # Tweak the visual presentation
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels_ = [str(int(int(l) / 1000)) for l in labels]
    ax.set_yticklabels(labels_)
    ax.set(ylabel="Count ($10^3$)", xlabel='Positions')
    f.savefig(save_dir + '/ud_class_dist.pdf')

    f, ax = plt.subplots(figsize=(7.25, 5.43))
    # Plot the orbital period with horizontal boxes
    sns.boxplot(data=ud_cls, x="position", y="POS tag",
                width=.4, palette="Set2", whis=[0, 100])
    # Tweak the visual presentation
    ax.set(ylabel="", xlabel='Positions')
    f.savefig(save_dir + '/ud.pdf')


def bias_experiment_k(dataset="conll03", model="bert-base-cased", experiment="position_bias"):
    model = f"{model.split('/')[-1]}"
    save_dir = os.path.join(plots_dir, experiment, dataset, model)
    os.makedirs(save_dir, exist_ok=True)
    if dataset in ["conll03", "ontonotes5"]:
        labels = NERDatasetbuilder.get_labels(dataset=dataset)
    else:
        labels = POSDatasetbuilder.get_labels(dataset=dataset)
    labels = list(np.unique([l.split("-")[-1] for l in labels]))
    labels.remove("O")
    reference = f"{model}-{experiment}-{dataset}"
    entity = "benamor"  # set to your entity and project
    runs = api.runs(entity + "/" + reference)

    summary_list, config_list, name_list = [], [], []
    runs_dfs = []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        if run.state == "finished":
            summary_list.append(run.summary._json_dict)
            df = run.history()
            dfs = []
            for k in range(1, 11):
                # k: duplication factor
                test_metrics_k = run.history(keys=[key for key in df.keys() if key.startswith(f"test/k={k}_")])
                test_metrics_k.columns = [col.split(f"={k}_")[-1] for col in test_metrics_k.columns]
                # test_metrics_k["k"] = k
                overall_batch = test_metrics_k.loc[:,
                                [key for key in test_metrics_k.keys() if not key.startswith(f"k=")]]

                overall_batch["batch_pos"] = 0
                overall_batch["batch_comp"] = 0
                batch_Positions = []
                for i in range(1, k + 1):
                    overall_batch_i = test_metrics_k.loc[:, [key for key in test_metrics_k.keys() if
                                                             key.startswith(f"k={i}.") and not key.startswith(
                                                                 f"k={i}.k=")]]
                    overall_batch_i.columns = [col.split(f"={i}.")[-1] for col in overall_batch_i.columns]
                    consistency_batch_i = test_metrics_k.loc[:,
                                          [key for key in test_metrics_k.keys() if key.startswith(f"k={i}.k=")]]
                    consistency_batch_i_ = []
                    for j in range(i, k + 1):
                        consistency_batch_i_j = consistency_batch_i.loc[:, [key for key in consistency_batch_i.keys() if
                                                                            key.startswith(f"k={i}.k={j}.")]]
                        consistency_batch_i_j.columns = [col.split(f"k={i}.k={j}.")[-1] for col in
                                                         consistency_batch_i_j.columns]
                        consistency_batch_i_j["batch_comp"] = j
                        consistency_batch_i_.append(consistency_batch_i_j)
                    consistency_batch_i = pd.concat(consistency_batch_i_)
                    consistency_batch_i.dtype = int
                    overall_batch_i = pd.concat([overall_batch_i, consistency_batch_i], axis=1)
                    overall_batch_i["batch_pos"] = i
                    batch_Positions.append(overall_batch_i)
                overall_batch_ = pd.concat([overall_batch] + batch_Positions)
                overall_batch_["k"] = k
                dfs.append(overall_batch_)
            run_results = pd.concat(dfs)
            run_results["run"] = run.name
            runs_dfs.append(run_results)
    df = pd.concat(runs_dfs)
    df.to_csv(os.path.join(save_dir, "results.csv"))

    return df


def bias_experiment_k_plot(df, experiment="position_bias"):
    save_dir = os.path.join(plots_dir, experiment)
    os.makedirs(save_dir, exist_ok=True)

    benchmarks = {"conll03": "CoNLL03", "ontonotes5": "OntoNotes5.0", "en_ewt": "UD_en", "tweebank": "TweeBank"}
    df["dataset"] = df["dataset"].map(benchmarks)

    # HeatMaps
    ## Batch Performance per alpha_k Table
    batch_f1_summary = df[df["batch_pos"] != 0][["model", "dataset", "overall_f1", "batch_pos"]].groupby(
        ["model", "dataset", "batch_pos"]).agg(
        [np.mean, np.std]) * 100
    batch_f1_summary.to_csv(os.path.join(save_dir, "batch_performance_summary.csv"))
    ## Overall Performance per k Table
    overall_f1_summary = df[df["batch_pos"] == 0][["model", "dataset", "overall_f1", "k"]].groupby(
        ["model", "dataset", "k"]).agg(
        [np.mean, np.std]) * 100
    overall_f1_summary.to_csv(os.path.join(save_dir, "overall_performance_summary.csv"))

    ## Batch Precision per alpha_k lineplot
    batch_precision_summary = df[df["batch_pos"] != 0][["model", "dataset", "overall_precision", "batch_pos"]].groupby(
        ["model", "dataset", "batch_pos"]).agg(
        [np.mean, np.std]) * 100
    batch_precision_summary = batch_precision_summary.reset_index()
    batch_precision_summary.columns = ["Model", "Dataset", "batch_pos", "precision", "std"]

    ## LinePlot
    f, ax = plt.subplots(figsize=(7.25, 5.43))
    sns.lineplot(
        data=df[df["batch_pos"] == df["batch_comp"]][df["batch_pos"] != 0][df["dataset"].isin(["OntoNotes5.0"])][
            df["model"].isin(["BERT", "BERT-Relative-Key", "Electra", "ERNIE"])], x="k", y="overall_f1",
        hue="model", style="dataset", markers=True,
        dashes=True, err_style="bars", errorbar="sd", palette="colorblind")

    ax.set_xlabel("Subset Position $\alpha_k$")
    ax.set_ylabel("$F1(\alpha_k)$")


    ## Consistency ratio (Correct agreement / all agreement)
    df_bert = df[df["model"] == "BERT"]
    df_bert["consistency"] = df.apply(lambda row: row["overall_correct"] / row["overall_total"], axis=1)
    correct_agreement = df_bert[df_bert["batch_pos"]==1].pivot_table(index="dataset", columns="batch_comp", values="overall_correct", aggfunc=np.mean)
    total_decisions = df_bert[df_bert["batch_pos"]==1].pivot_table(index="dataset", columns="batch_comp", values="overall_total", aggfunc=np.mean)
    for i in range(2, 11):
        total_decisions[i] = total_decisions[1]

    tp_1 = correct_agreement[1]
    tp_ratio = correct_agreement.copy()
    for i in range(1, 11):
        tp_ratio[i] = tp_ratio[i] / tp_1
    consistency_mean = correct_agreement / total_decisions
    benchmarks = {"conll03": "CoNLL03", "ontonotes5": "OntoNotes5.0", "en_ewt": "UD_en", "tweebank": "TweeBank"}
    f, ax = plt.subplots(figsize=(5, 3.75))
    sns.heatmap(data=consistency_mean * 100, annot=True, fmt=".2f", cmap="Blues", square=True,
                robust=True, cbar_kws={"shrink": .5}, linewidths=.5,
                annot_kws={"fontsize": 'xx-small', "fontstretch": 'extra-condensed'}, ax=ax)

    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels_ = [benchmarks[l] for l in labels]
    ax.set_yticklabels(labels_)
    ax.set(ylabel=r"", xlabel=r"$k$")
    ax.minorticks_off()
    ax.tick_params(bottom=False, left=False)
    f.savefig(os.path.join(save_dir, 'heatmap_batch_consistency_pos.pdf'))
    plt.close()

    ## Consistency ratio (Correct agreement / correct agreement) lineplot
    dfss = []
    cos_mean = tp_ratio.T
    for col in cos_mean.columns:
        ss = cos_mean[col]
        dfs = pd.DataFrame(ss)
        dfs["k"] = [i for i in range(1, 11)]
        dfs["dataset"] = col
        dfs.columns = ["consistency", "k", "dataset"]
        dfss.append(dfs)
    cos_batch_k = pd.concat(dfss)
    cos_batch_k["dataset"] = cos_batch_k["dataset"].map(benchmarks)
    f, ax = plt.subplots(figsize=(5, 3.75))
    sns.lineplot(data=cos_batch_k, x="k", y="consistency", ax=ax, hue="dataset", palette="colorblind", style="dataset",
                 markers=["o", "o", "o", "o"])
    sns.move_legend(ax, "upper left", bbox_to_anchor=(0, 0.4), title="Dataset",
                    fontsize="x-small")
    ax.set(ylabel=r'$consistency(1,k)$', xlabel=r'$k$')
    f.savefig(os.path.join(save_dir, 'lines_batch_consistency_pos.pdf'))
    plt.close()


    # Correct and total agreement per class
    if dataset in ["conll03", "ontonotes"]:
        labels = NERDatasetbuilder.get_labels(dataset=dataset)
    else:
        labels = POSDatasetbuilder.get_labels(dataset=dataset)
    labels = list(np.unique([l.split("-")[-1] for l in labels]))
    labels.remove("O")
    for cls in labels:
        correct_batch_pos_mean = df.pivot_table(index="batch_comp", columns="batch_pos", values=f"{cls}.correct",
                                                aggfunc=np.mean)
        matrix = correct_batch_pos_mean.values
        upper_indices = np.triu_indices(matrix.shape[1])
        matrix[upper_indices] = matrix.T[upper_indices]
        correct_batch_pos_mean.iloc[:, :] = matrix
        correct_batch_pos_std = df.pivot_table(index="batch_comp", columns="batch_pos", values=f"{cls}.correct",
                                               aggfunc=np.std)
        matrix = correct_batch_pos_std.values
        matrix[upper_indices] = matrix.T[upper_indices]
        correct_batch_pos_std.iloc[:, :] = matrix
        f, ax = plt.subplots(figsize=(5, 3.75))
        sns.heatmap(data=correct_batch_pos_mean, annot=correct_batch_pos_std.values, robust=True,
                    annot_kws={"fontsize": 'xx-small', "fontstretch": 'extra-condensed'}, ax=ax)
        # locs, labels = plt.xticks()
        # labels[0] = plt.Text(0.5, 0, "all")
        # plt.xticks(locs, labels)
        ax.set(ylabel=r"$Test_{subset}(\alpha_{[1,k]})$", xlabel=r"$Test_{subset}(\alpha_{[1,k]})$")
        f.savefig(os.path.join(save_dir, f'{cls}_heatmap_correct_pos.pdf'))
        plt.close()

        correct_batch_pos_mean = df.pivot_table(index="batch_comp", columns="batch_pos", values=f"{cls}.total",
                                                aggfunc=np.mean)
        matrix = correct_batch_pos_mean.values
        upper_indices = np.triu_indices(matrix.shape[1])
        matrix[upper_indices] = matrix.T[upper_indices]
        correct_batch_pos_mean.iloc[:, :] = matrix
        correct_batch_pos_std = df.pivot_table(index="batch_comp", columns="batch_pos", values=f"{cls}.total",
                                               aggfunc=np.std)
        matrix = correct_batch_pos_std.values
        matrix[upper_indices] = matrix.T[upper_indices]
        correct_batch_pos_std.iloc[:, :] = matrix
        f, ax = plt.subplots(figsize=(5, 3.75))
        sns.heatmap(data=correct_batch_pos_mean, annot=correct_batch_pos_std.values, robust=True,
                    annot_kws={"fontsize": 'xx-small', "fontstretch": 'extra-condensed'}, ax=ax)
        # locs, labels = plt.xticks()
        # labels[0] = plt.Text(0.5, 0, "all")
        # plt.xticks(locs, labels)
        ax.set(ylabel=r"$Test_{subset}(\alpha_{[1,k]})$", xlabel=r"$Test_{subset}(\alpha_{[1,k]})$")
        f.savefig(os.path.join(save_dir, f'{cls}_heatmap_correct_pos.pdf'))
        plt.close()

    # Line Plots
    ## F1 performance of all classes per batch position
    for cls in labels:
        batch_number = df[df["batch_pos"] != 0][f"{cls}.number"].unique()[0]
        at = AnchoredText(
            f"Support: {batch_number}", prop=dict(size=15), frameon=True, loc='upper right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        # Per position
        f, ax = plt.subplots(figsize=(3.54, 2.65))
        sns.pointplot(data=df[df["batch_pos"] != 0], x="batch_pos", y=f"{cls}.f1", errorbar="sd", markers="x",
                      errwidth=1, scale=0.5, palette="Paired", hue="k", linestyles="--")
        # Tweak the visual presentation
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=r"$k$", fontsize="xx-small")
        ax.add_artist(at)
        ax.set(ylabel=f"F1({cls})", xlabel='Batch(k) Position')
        # Per k factor
        f.savefig(os.path.join(save_dir, f'{cls}_f1_pos.pdf'))
        f, ax = plt.subplots(figsize=(3.54, 2.65))
        at = AnchoredText(
            f"Support: {batch_number}", prop=dict(size=15), frameon=True, loc='upper right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        sns.pointplot(data=df[df["batch_pos"] != 0], x="k", y=f"{cls}.f1", errorbar="sd", markers="x",
                      errwidth=1, scale=0.5, palette="Paired", hue="batch_pos", linestyles="--")
        # Tweak the visual presentation
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=r"$Test_{subset}(\alpha_{[1,k]})$",
                        fontsize="xx-small")
        ax.set(ylabel=f"F1({cls})", xlabel=r'$k$')
        ax.add_artist(at)
        f.savefig(os.path.join(save_dir, f'{cls}_f1_k.pdf'))
    plt.close()

    ## F1 performance of all classes per k across all positions
    class_dfs = []
    for cls in labels:
        cls_df = df.loc[:,
                 [key for key in df.keys() if key.startswith(f"{cls}.")] + ["k", "run", "batch_pos",
                                                                            "batch_comp"]]
        cols = [col.split(".")[-1] if col.startswith(f"{cls}.") else col for col in cls_df.columns]
        cls_df.columns = cols
        cls_df["class"] = cls
        class_dfs.append(cls_df)
    class_df = pd.concat(class_dfs, ignore_index=True)
    overall_class_perf = class_df[class_df["batch_pos"] == 0]
    f, ax = plt.subplots(figsize=(3.54, 2.65))
    sns.lineplot(data=overall_class_perf, x="k", y="f1", errorbar="sd", hue="class", ax=ax)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Class",
                    fontsize="xx-small")
    ax.set(ylabel=f"F1", xlabel=r'$k$')
    f.savefig(os.path.join(save_dir, f'allclasses_f1_k.pdf'))
    plt.close()

    ## F1 per class/k
    for cls in labels:
        batch_number = overall_class_perf[overall_class_perf["class"] == cls][f"number"].unique()
        legend = "\n".join([f"k={i}: {n}" for i, n in enumerate(batch_number)])
        f, ax = plt.subplots(figsize=(3.54, 2.65))
        sns.lineplot(data=overall_class_perf[overall_class_perf["class"] == cls], x="k", y=f"f1", errorbar="sd",
                     )
        ax.set(ylabel=f"F1({cls})", xlabel=r'$k$')
        at = AnchoredText(
            f"Support\n{legend}", prop=dict(size=6), frameon=True, loc='upper right', bbox_to_anchor=(1, 1),
            bbox_transform=ax.transAxes)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        f.savefig(os.path.join(save_dir, f'{cls}_f1_k_nopos.pdf'))
    plt.close()

    # HeatMap of classes performance / batch position
    class_batch_pos_mean = class_df.pivot_table(index="class", columns="batch_pos", values="f1", aggfunc=np.mean)
    class_batch_pos_std = class_df.pivot_table(index="class", columns="batch_pos", values="f1", aggfunc=np.std)
    f, ax = plt.subplots(figsize=(5, 3.75))
    sns.heatmap(data=class_batch_pos_std * 100, annot=class_batch_pos_mean.values * 100, fmt=".2f", robust=True,
                annot_kws={"fontsize": 'xx-small', "fontstretch": 'extra-condensed'}, ax=ax)
    ax.set(xticklabels=(["all"] + [str(i) for i in range(1, 11)]))
    ax.set(ylabel="Class", xlabel=r"$Test_{subset}(\alpha_{[1,k]})$")
    ax.xaxis.tick_top()
    ax.minorticks_off()
    f.savefig(os.path.join(save_dir, 'heatmap_class_f1_pos.pdf'))
    # Heatmap of classes / k
    class_batch_pos_mean = class_df[class_df["batch_pos"] == 0].pivot_table(index="class", columns="k", values="f1",
                                                                            aggfunc=np.mean)
    class_batch_pos_std = class_df[class_df["batch_pos"] == 0].pivot_table(index="class", columns="k", values="f1",
                                                                           aggfunc=np.std)
    f, ax = plt.subplots(figsize=(5, 3.75))
    sns.heatmap(data=class_batch_pos_std * 100, annot=class_batch_pos_mean.values * 100, fmt=".2f", robust=True,
                annot_kws={"fontsize": 'xx-small', "fontstretch": 'extra-condensed'}, ax=ax)
    # ax.set(xticklabels=(["all"] + [str(i) for i in range(1, 11)]))
    ax.set(ylabel="Class", xlabel=r"$k$")
    ax.xaxis.tick_top()
    ax.minorticks_off()
    f.savefig(os.path.join(save_dir, 'heatmap_class_f1_k.pdf'))
    plt.close()


def emb_analysis(dataset="conll03"):
    experiment = "bert_position_bias_eval"
    save_dir = os.path.join(plots_dir, experiment, dataset)
    os.makedirs(save_dir, exist_ok=True)
    labels = NERDatasetbuilder.get_labels(dataset=dataset)
    labels = list(np.unique([l.split("-")[-1] for l in labels]))
    labels.remove("O")
    entity = "benamor"  # set to your entity and project
    runs = api.runs(entity + "/" + experiment + "-" + dataset)

    df = []
    df1 = []
    general_info = []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        if run.state == "finished":
            pos_df = []
            word_df = []
            seq_path = wandb.restore("seq_info.pt", run_path="/".join(run.path), root=os.path.join(save_dir, run.id))
            seq_info = torch.load(seq_path.name)
            pos_cos_path = wandb.restore("pos_cos.pt", run_path="/".join(run.path), root=os.path.join(save_dir, run.id))
            word_cos_path = wandb.restore("word_cos.pt", run_path="/".join(run.path),
                                          root=os.path.join(save_dir, run.id))
            pos_embeddings = torch.load(pos_cos_path.name)
            word_embeddings = torch.load(word_cos_path.name)
            for i, embs in enumerate(pos_embeddings):
                for k, emb_cos in embs.items():
                    first_token = emb_cos[0, :]
                    avg_token = emb_cos.mean(axis=0)
                    for j, cos_sim in enumerate(first_token):
                        pos_df.append([i, k.split("=")[-1], j, cos_sim, avg_token[j]])
            for i, embs in enumerate(word_embeddings):
                for k, emb_cos in embs.items():
                    first_token = emb_cos[0, :]
                    avg_token = emb_cos.mean(axis=0)
                    for j, cos_sim in enumerate(first_token):
                        word_df.append([i, k.split("=")[-1], j, cos_sim, avg_token[j]])

            pos_df = pd.DataFrame(pos_df, columns=["id", "batch_pos", "layer", "first_token", "avg_sim"])
            pos_df['run'] = run.name
            word_df = pd.DataFrame(word_df, columns=["id", "batch_pos", "layer", "first_token", "avg_sim"])
            word_df['run'] = run.name
            df.append(pos_df)
            df1.append(word_df)

    ## First token Embeddings plot
    pos_df = pd.concat(df)
    word_df = pd.concat(df1)

    f, ax = plt.subplots(figsize=(5, 3.75))
    sns.lineplot(data=pos_df[pos_df["layer"] == 12], x="batch_pos", y="first_token", ax=ax, label="position embeddings")
    sns.lineplot(data=word_df[word_df["layer"] == 12], x="batch_pos", y="first_token", ax=ax, label="word embeddings")

    ax.set(ylabel=f"Cosine Similarity", xlabel='Batch(k) Position')
    # Per k factor
    f.savefig(os.path.join(save_dir, f'pos_cos_first_token.pdf'))

    ## Avg Embeddings plot
    f, ax = plt.subplots(figsize=(5, 3.75))
    sns.lineplot(data=pos_df[pos_df["layer"] == 12], x="batch_pos", y="avg_sim", ax=ax, label="position embeddings")
    sns.lineplot(data=word_df[word_df["layer"] == 12], x="batch_pos", y="avg_sim", ax=ax, label="word embeddings")

    ax.set(ylabel=f"Cosine Similarity", xlabel='Batch(k) Position')
    # Per k factor
    f.savefig(os.path.join(save_dir, f'pos_cos_avg_token.pdf'))


def get_results(models, datasets, experiment="position_bias"):
    model_names = ["BERT", "Electra", "ERNIE", "BERT-Relative-Key", "BERT-Relative-Key-Query", "Bloom", "GPT2",
                   "GPT2-Large"]
    dfs = []
    for dataset in datasets:
        dffs = []
        for i,model in enumerate(models):
            df = bias_experiment_k(model=model, dataset=dataset, experiment=experiment)
            df["model"] = model_names[i]
            dffs.append(df)
        dff = pd.concat(dffs)
        dff["dataset"] = dataset
        dfs.append(dff)
    df = pd.concat(dfs)
    df.to_csv(os.path.join(plots_dir, f"bert_{experiment}_eval.csv"))
    return df

if __name__ == "__main__":
    # dataset_plot()
    # models = ["bert-base-uncased", "google/electra-base-discriminator", "nghuyong/ernie-2.0-base-en",
    #           "zhiheng-huang/bert-base-uncased-embedding-relative-key",
    #           "zhiheng-huang/bert-base-uncased-embedding-relative-key-query", "bigscience/bloom-560m", "gpt2",
    #           "gpt2-large"]
    # datasets = ["conll03", "ontonotes5", "en_ewt", "tweebank"]
    # pos_bias = get_results(models, datasets, experiment="position_bias")
    pos_bias = pd.read_csv(os.path.join(plots_dir, "bert_position_bias_eval.csv"))
    bias_experiment_k_plot(pos_bias, experiment="position_bias")

    models = ["bert-base-uncased"]
    datasets = ["conll03", "ontonotes5", "en_ewt", "tweebank"]
    # finetune_shift = get_results(models, datasets, experiment="finetune_shift")
    # finetune_concat = get_results(models, datasets, experiment="finetune_concat")
    # finetune_shift = pd.read_csv(os.path.join(plots_dir, "bert_finetune_shift_eval.csv"))
    # finetune_concat = pd.read_csv(os.path.join(plots_dir, "bert_finetune_concat_eval.csv"))
    # bias_experiment_k_plot(finetune_shift, experiment="finetune_shift")
    # bias_experiment_k_plot(finetune_concat, experiment="finetune_concat")

    # emb_analysis(dataset="conll03")
