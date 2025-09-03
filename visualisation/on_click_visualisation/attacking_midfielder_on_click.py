import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you prefer
import matplotlib.pyplot as plt
from .default_on_click import plot_stat_percentile_bar
from config import stats_dict
def attacking_midfielder_on_click(df, df2, label):
    relevant_columns = ['ShT/90','Gls/90','Poss Won/90','OP-KP/90','Hdr %','Ch C/90','Asts/90','Sprints/90','Drb/90']
    df = df.dropna(subset=relevant_columns)
    df2 = df2.dropna(subset=relevant_columns)
    fig2, ax2 = plt.subplots()

    for column in relevant_columns:
        plot_stat_percentile_bar(df, df2, column, label, stats_dict[column])
    ax2.set_yticks(range(0, 101, 10))

    plt.xticks(rotation=90)

    plt.title(label, color='white')
    plt.tight_layout()
    plt.setp(ax2.spines.values(), color="white")
    plt.ylabel('Percentile', color='white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.xaxis.label.set_color('white')
    ax2.set_facecolor('#333333')
    fig2.set_facecolor('#333333')
    plt.show(block=True)