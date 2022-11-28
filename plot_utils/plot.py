# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import os

from matplotlib.offsetbox import AnchoredText

from dataset.ner_dataset import NERDataset, NERDatasetbuilder

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
plots_dir = 'plots/files/'
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

    # Datasets statistics
    conll03_seq = pd.DataFrame(conll03.sequence_lengths, columns=["seq_lengths"])
    f, ax_hist = plt.subplots(1, figsize=(3.54, 2.65))
    sns.histplot(conll03_seq["seq_lengths"], ax=ax_hist,
                 edgecolor='black')

    ax_hist.set(xlabel='Sequence lengths', ylabel='Count')

    plt.legend()
    f.savefig(save_dir + 'conll03_seq_lengths.pdf')

    ontonotes5_seq = pd.DataFrame(ontonotes5.sequence_lengths, columns=["seq_lengths"])
    f, ax_hist = plt.subplots(1, figsize=(3.54, 2.65))
    sns.histplot(ontonotes5_seq["seq_lengths"], ax=ax_hist,
                 edgecolor='black')

    ax_hist.set(xlabel='Sequence lengths', ylabel='Count')

    plt.legend()
    f.savefig(save_dir + 'ontonotes5_seq_lengths.pdf')

    df = pd.DataFrame(
        [(x, "conll03") for x in conll03.sequence_lengths] + [(x, "ontonotes5") for x in ontonotes5.sequence_lengths],
        columns=["seq_lengths", "dataset"])

    f, ax = plt.subplots(figsize=(7.25, 2.43))

    # Plot the orbital period with horizontal boxes
    sns.boxplot(x="seq_lengths", y="dataset", data=df,
                width=.3, palette="vlag", whis=[0, 100])

    # Tweak the visual presentation
    ax.set(ylabel="", xlabel='Sequence lengths')
    f.savefig(save_dir + 'seq_lengths.pdf')

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

    f, ax = plt.subplots(figsize=(3.54, 2.65))
    # Plot the orbital period with horizontal boxes
    sns.boxplot(data=dff, x="position", y="class",
                width=.4, palette="Set2", whis=[0, 100])
    # Tweak the visual presentation
    ax.set(ylabel="", xlabel='Positions')
    f.savefig(save_dir + 'conll03.pdf')

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
    dff = pd.DataFrame(dfs, columns=["position", "class"])

    f, ax = plt.subplots(figsize=(7.25, 5.43))
    # Plot the orbital period with horizontal boxes
    sns.boxplot(data=dff, x="position", y="class",
                width=.4, palette="Set2", whis=[0, 100])
    # Tweak the visual presentation
    ax.set(ylabel="", xlabel='Positions')
    f.savefig(save_dir + 'ontonotes.pdf')


def bias_experiment(experiment="bert_position_bias_synthetic", dataset="ontonotes5"):
    save_dir = os.path.join(plots_dir, experiment, dataset)
    os.makedirs(save_dir, exist_ok=True)
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
            for k in range(1, 11):
                df_ = run.history(keys=[key for key in df.keys() if key.startswith(f"test/k={k}_")])
                df_.columns = [col.split(f"={k}_")[-1] for col in df_.columns]
                df_["k"] = k
                dfs.append(df_)

            results = pd.concat(dfs)

            f, ax = plt.subplots(figsize=(3.54, 2.65))
            # Plot the orbital period with horizontal boxes
            sns.lineplot(data=results, x="k", y="overall_f1",
                         palette="Set2", markers=True)
            # Tweak the visual presentation
            ax.set(ylabel="F1", xlabel='k')
            f.savefig(save_dir + f'{dataset}_f1.jpg')
            for cls in labels:
                if cls != "O":
                    f, ax = plt.subplots(figsize=(3.54, 2.65))
                    # Plot the orbital period with horizontal boxes
                    sns.lineplot(data=results, x="k", y=f"{cls}.f1",
                                 palette="Set2", markers=True)
                    # Tweak the visual presentation
                    ax.set(ylabel=f"F1({cls})", xlabel='k')
                    f.savefig(save_dir + f'{dataset}_{cls}_f1.jpg')


def bias_experiment_k(dataset="conll03"):
    experiment = "bert_position_bias_no_cv"
    save_dir = os.path.join(plots_dir, experiment, dataset)
    os.makedirs(save_dir, exist_ok=True)
    labels = NERDatasetbuilder.get_labels(dataset=dataset)
    labels = list(np.unique([l.split("-")[-1] for l in labels]))
    labels.remove("O")
    entity = "benamor"  # set to your entity and project
    runs = api.runs(entity + "/" + experiment + "-" + dataset)

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

    # HeatMaps
    ## Batch Performance per k
    ### F1 measure
    perf_batch_pos_mean = df.pivot_table(index="k", columns="batch_pos", values="overall_f1", aggfunc=np.mean)
    matrix = perf_batch_pos_mean.iloc[:, 1:].values
    upper_indices = np.triu_indices(matrix.shape[1])
    matrix[upper_indices] = matrix.T[upper_indices]
    perf_batch_pos_mean.iloc[:, 1:] = matrix
    perf_batch_pos_std = df.pivot_table(index="k", columns="batch_pos", values="overall_f1", aggfunc=np.std)
    matrix = perf_batch_pos_std.iloc[:, 1:].values
    matrix[upper_indices] = matrix.T[upper_indices]
    perf_batch_pos_std.iloc[:, 1:] = matrix
    f, ax = plt.subplots(figsize=(5, 3.75))
    sns.heatmap(data=perf_batch_pos_mean * 100, annot=perf_batch_pos_std.values * 100, fmt=".2f", robust=True,
                annot_kws={"fontsize": 'xx-small', "fontstretch": 'extra-condensed'}, ax=ax)
    ax.set(xticklabels=(["all"] + [str(i) for i in range(1, 11)]))
    ax.set(ylabel="$k$", xlabel=r"$Test_{subset}(\alpha_{[1,k]})$")
    ax.xaxis.tick_top()
    ax.minorticks_off()
    f.savefig(os.path.join(save_dir, 'heatmap_batch_f1_pos.pdf'))
    plt.close()

    ### Precision and Recall
    precision_batch_pos_mean = df.pivot_table(index="k", columns="batch_pos", values="overall_precision",
                                              aggfunc=np.mean)
    matrix = precision_batch_pos_mean.iloc[:, 1:].values
    upper_indices = np.triu_indices(matrix.shape[1])
    matrix[upper_indices] = matrix.T[upper_indices]
    precision_batch_pos_mean.iloc[:, 1:] = matrix
    precision_batch_pos_std = df.pivot_table(index="k", columns="batch_pos", values="overall_precision", aggfunc=np.std)
    matrix = precision_batch_pos_std.iloc[:, 1:].values
    matrix[upper_indices] = matrix.T[upper_indices]
    precision_batch_pos_std.iloc[:, 1:] = matrix
    f, ax = plt.subplots(figsize=(5, 3.75))
    sns.heatmap(data=precision_batch_pos_mean * 100, annot=precision_batch_pos_std.values * 100, fmt=".2f", robust=True,
                annot_kws={"fontsize": 'xx-small', "fontstretch": 'extra-condensed'}, ax=ax)
    ax.set(xticklabels=(["all"] + [str(i) for i in range(1, 11)]))
    ax.set(ylabel="$k$", xlabel=r"$Test_{subset}(\alpha_{[1,k]})$")
    ax.xaxis.tick_top()
    ax.minorticks_off()
    f.savefig(os.path.join(save_dir, 'heatmap_batch_precision_pos.pdf'))
    # ---
    recall_batch_pos_mean = df.pivot_table(index="k", columns="batch_pos", values="overall_recall", aggfunc=np.mean)
    matrix = recall_batch_pos_mean.iloc[:, 1:].values
    upper_indices = np.triu_indices(matrix.shape[1])
    matrix[upper_indices] = matrix.T[upper_indices]
    recall_batch_pos_mean.iloc[:, 1:] = matrix
    recall_batch_pos_std = df.pivot_table(index="k", columns="batch_pos", values="overall_recall", aggfunc=np.std)
    matrix = recall_batch_pos_std.iloc[:, 1:].values
    matrix[upper_indices] = matrix.T[upper_indices]
    recall_batch_pos_std.iloc[:, 1:] = matrix
    f, ax = plt.subplots(figsize=(5, 3.75))
    sns.heatmap(data=recall_batch_pos_mean * 100, annot=recall_batch_pos_std.values * 100, fmt=".2f", robust=True,
                annot_kws={"fontsize": 'xx-small', "fontstretch": 'extra-condensed'}, ax=ax)
    ax.set(xticklabels=(["all"] + [str(i) for i in range(1, 11)]))
    ax.set(ylabel="$k$", xlabel=r"$Test_{subset}(\alpha_{[1,k]})$")
    ax.xaxis.tick_top()
    ax.minorticks_off()
    f.savefig(os.path.join(save_dir, 'heatmap_batch_recall_pos.pdf'))
    plt.close()

    ## Consistency ratio (Correct agreement / all agreement)
    df["consistency"] = df.apply(lambda row: row["overall_correct"] / row["overall_total"], axis=1)
    consistency_batch_pos_mean = df.pivot_table(index="batch_comp", columns="batch_pos", values="consistency",
                                                aggfunc=np.mean)
    matrix = consistency_batch_pos_mean.values
    upper_indices = np.triu_indices(matrix.shape[1])
    matrix[upper_indices] = matrix.T[upper_indices]
    consistency_batch_pos_mean.iloc[:, :] = matrix
    consistency_batch_pos_std = df.pivot_table(index="batch_comp", columns="batch_pos", values="consistency",
                                               aggfunc=np.std)
    matrix = consistency_batch_pos_std.values
    matrix[upper_indices] = matrix.T[upper_indices]
    consistency_batch_pos_std.iloc[:, :] = matrix
    f, ax = plt.subplots(figsize=(5, 3.75))
    sns.heatmap(data=consistency_batch_pos_mean * 100, annot=consistency_batch_pos_std.values * 100, fmt=".2f",
                robust=True,
                annot_kws={"fontsize": 'xx-small', "fontstretch": 'extra-condensed'}, ax=ax)
    # locs, labels = plt.xticks()
    # labels[0] = plt.Text(0.5, 0, "all")
    # plt.xticks(locs, labels)
    ax.set(ylabel=r"$Test_{subset}(\alpha_{[1,k]})$", xlabel=r"$Test_{subset}(\alpha_{[1,k]})$")
    ax.xaxis.tick_top()
    ax.minorticks_off()
    f.savefig(os.path.join(save_dir, 'heatmap_batch_consistency_pos.pdf'))
    plt.close()

    ## Correct and total agreements consistency
    correct_batch_pos_mean = df.pivot_table(index="batch_comp", columns="batch_pos", values="overall_correct",
                                            aggfunc=np.mean)
    matrix = correct_batch_pos_mean.values
    upper_indices = np.triu_indices(matrix.shape[1])
    matrix[upper_indices] = matrix.T[upper_indices]
    correct_batch_pos_mean.iloc[:, :] = matrix
    correct_batch_pos_std = df.pivot_table(index="batch_comp", columns="batch_pos", values="overall_correct",
                                           aggfunc=np.std)
    matrix = correct_batch_pos_std.values
    matrix[upper_indices] = matrix.T[upper_indices]
    correct_batch_pos_std.iloc[:, :] = matrix
    f, ax = plt.subplots(figsize=(5, 3.75))
    sns.heatmap(data=correct_batch_pos_mean, annot=correct_batch_pos_std.values, robust=True,
                annot_kws={"fontsize": 'xx-small', "fontstretch": 'extra-condensed'}, ax=ax)
    ax.set(ylabel=r"$Test_{subset}(\alpha_{[1,k]})$", xlabel=r"$Test_{subset}(\alpha_{[1,k]})$")
    ax.xaxis.tick_top()
    ax.minorticks_off()
    f.savefig(os.path.join(save_dir, 'heatmap_batch_correct_pos.pdf'))

    total_batch_pos_mean = df.pivot_table(index="batch_comp", columns="batch_pos", values="overall_total",
                                          aggfunc=np.mean)
    matrix = total_batch_pos_mean.values
    upper_indices = np.triu_indices(matrix.shape[1])
    matrix[upper_indices] = matrix.T[upper_indices]
    total_batch_pos_mean.iloc[:, :] = matrix
    total_batch_pos_std = df.pivot_table(index="batch_comp", columns="batch_pos", values="overall_total",
                                         aggfunc=np.std)
    matrix = total_batch_pos_std.values
    matrix[upper_indices] = matrix.T[upper_indices]
    total_batch_pos_std.iloc[:, :] = matrix
    f, ax = plt.subplots(figsize=(5, 3.75))
    sns.heatmap(data=total_batch_pos_mean, annot=total_batch_pos_std.values, robust=True,
                annot_kws={"fontsize": 'xx-small', "fontstretch": 'extra-condensed'}, ax=ax)
    ax.set(ylabel=r"$Test_{subset}(\alpha_{[1,k]})$", xlabel=r"$Test_{subset}(\alpha_{[1,k]})$")
    ax.xaxis.tick_top()
    ax.minorticks_off()
    f.savefig(os.path.join(save_dir, 'heatmap_batch_total_pos.pdf'))
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
        f.savefig(os.path.join(save_dir ,f'{cls}_f1_pos.pdf'))
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
        f.savefig(os.path.join(save_dir ,f'{cls}_f1_k.pdf'))
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
    class_batch_pos_mean = class_df[class_df["batch_pos"]==0].pivot_table(index="class", columns="k", values="f1", aggfunc=np.mean)
    class_batch_pos_std = class_df[class_df["batch_pos"]==0].pivot_table(index="class", columns="k", values="f1", aggfunc=np.std)
    f, ax = plt.subplots(figsize=(5, 3.75))
    sns.heatmap(data=class_batch_pos_std * 100, annot=class_batch_pos_mean.values * 100, fmt=".2f", robust=True,
                annot_kws={"fontsize": 'xx-small', "fontstretch": 'extra-condensed'}, ax=ax)
    # ax.set(xticklabels=(["all"] + [str(i) for i in range(1, 11)]))
    ax.set(ylabel="Class", xlabel=r"$k$")
    ax.xaxis.tick_top()
    ax.minorticks_off()
    f.savefig(os.path.join(save_dir, 'heatmap_class_f1_k.pdf'))
    plt.close()


if __name__ == "__main__":
    # dataset_plot()
    # bias_experiment()
    # bias_experiment_k(dataset="conll03")
    bias_experiment_k(dataset="ontonotes5")
