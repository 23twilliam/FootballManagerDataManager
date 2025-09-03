import numpy as np
import matplotlib
import mplcursors
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you prefer
import matplotlib.pyplot as plt
import visualisation.on_click_visualisation
import math
def plotting(avg, df, valMax, plot_type):
    mask = df['Transfer Value'] > valMax
    df2 = df[~mask]
    avg.columns = ['avg']
    r = np.nanmean(avg)
    df2.loc[:, 'avg'] = avg.values
    df2 = df2.sort_values(by='avg', ascending=False)
    fig, ax = plt.subplots()
    df2 = df2.head(min(int(math.sqrt(len(df2)) * 25), 1500))
    x = df2['Transfer Value'].values
    y = df2['avg'].values
    labels = df2['Name'].values

    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }

    cmap = plt.get_cmap('RdYlGn')
    norm = plt.Normalize(y.min(), y.max())
    line_colors = cmap(norm(y))


    s = max(10, 50 - len(df2) // 100)
    scatter = plt.scatter(x, y, s=s, c=line_colors, marker='.')

    plt.ylim(0, (max(avg) + max(avg) / 20))
    plt.tight_layout()

    ax.set_facecolor('#333333')
    fig.set_facecolor('#333333')

    c2 = mplcursors.cursor(hover=True)
    c3 = mplcursors.cursor()
    plt.setp(ax.spines.values(), color="white")
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Transfer Value')
    ax.xaxis.label.set_color('white')

    scatter._player_labels = labels  # Store individual labels as custom attribute
    print(f"Number Of Players: {len(df2)}")
    @c2.connect("add")
    def _(sel):
        sel.annotation.get_bbox_patch().set(color="#FFFFFF", ec="#999999", snap=True, boxstyle="round,pad=0.3")
        sel.annotation.arrow_patch.set(visible=False)

        # The correct way to get the index with mplcursors
        index = sel.index  # This is the key - use sel.index, not sel.target.index
        label = labels[index]
        sel.annotation.set_text(label)

    @c3.connect("add")
    def _1(sel):
        index = sel.index  # Again, use sel.index
        label = labels[index]


        plt.close(fig)
        match plot_type:
            case 1:
                visualisation.on_click_visualisation.keeper_on_click(df, df2, label)
            case 2:
                visualisation.on_click_visualisation.defender_on_click(df, df2, label)
            case 3:
                visualisation.on_click_visualisation.fullback_on_click(df, df2, label)
            case 4:
                visualisation.on_click_visualisation.wingback_on_click(df, df2, label)
            case 5:
                visualisation.on_click_visualisation.defensive_midfielder_on_click(df, df2, label)
            case 6:
                visualisation.on_click_visualisation.central_midfielder_on_click(df, df2, label)
            case 7:
                visualisation.on_click_visualisation.attacking_midfielder_on_click(df, df2, label)
            case 8:
                visualisation.on_click_visualisation.winger_on_click(df, df2, label)
            case 9:
                visualisation.on_click_visualisation.striker_on_click(df, df2, label)
            case _:
                print("error")
    plt.show(block=True)


# Looking at your code, here are the main issues that need addressing:
# Critical Problems:

# Dangerous data modification: df2.loc[:, 'avg'] = avg.values modifies the original DataFrame if it's a view. This could corrupt your data unexpectedly. Always use df2 = df2.copy() before modifying.
# Silent failures everywhere: Your code has zero error handling. What happens if avg and df have different lengths? If valMax is invalid? If the visualization module fails to import? Your program will crash with cryptic errors.
# Terrible variable names: df2, c2, c3, _, _1, r - these tell me nothing about what they represent. Code should be self-documenting.
# Hard-coded magic numbers: 25, 1500, 100, 50, 10, 20 - scattered throughout with no explanation of what they represent or why those specific values were chosen.
#
# Design Issues:
#
# Function does too much: This single function handles filtering, sorting, plotting, styling, event handling, and navigation. It should be broken into smaller, focused functions.
# Tight coupling: The function is hardwired to specific column names ('Transfer Value', 'Name') and a specific visualization module structure. This makes it brittle and hard to reuse.
# Inconsistent data handling: You use np.nanmean(avg) but never use the result (r). You sort by 'avg' but the sorting logic seems disconnected from the actual plotting purpose.
# Resource management: You're creating matplotlib objects but not properly managing their lifecycle, which can lead to memory leaks in longer-running applications.
#
# Code Quality Issues:
#
# Poor parameter validation: No checks if inputs are valid, have expected columns, or are the right data types.
# Mixing responsibilities: UI styling, data processing, and business logic are all jumbled together.
#
# Suggested improvements: Extract data filtering, add proper error handling, use descriptive names, create configuration objects for styling, separate the event handlers into their own class, and add input validation. The core logic is sound, but the implementation needs significant refactoring to be maintainable and reliable.