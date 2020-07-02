import matplotlib.pyplot as plt
import seaborn as sns


def classification_heatmap(classification_df):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(classification_df.clip(lower=-40000),
                center=0,
                cmap="YlGnBu",
                yticklabels=classification_df.columns,
                ax=ax)
    ax.set_title('Ability to detect correct time series')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
    ax.set_xlabel('True labels')
    ax.set_ylabel('Scored labels')


def highlight_max(s):
    is_max = s == s[: -1].max()
    return ['background-color: yellow' if v else '' for v in is_max]

