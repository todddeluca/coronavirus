
import datetime

import matplotlib.pyplot as plt
import numpy as np

from .data import fill_before_first, make_days_since, before_threshold
from .util import fit_power_law


# Seasonal Flu Death Source
# https://www.cdc.gov/flu/about/burden/index.html
# seasons: 2010-2011, 2011-2012, 2012-2013, 2013-2014, 2014-2015, 2015-2016, 2016-2017, 2017-2018
seasonal_flu_deaths = np.array([37000, 12000, 43000, 38000, 51000, 23000, 38000, 61000])
# US Population Source
# https://www.worldometers.info/world-population/us-population/
us_pop = 331000000  # estimated 2020 population
seasonal_flu_dpm = seasonal_flu_deaths.mean() / us_pop * 1e6
bad_flu_dpm = seasonal_flu_deaths.max() / us_pop * 1e6
# print(f'seasonal_flu_dpm: {seasonal_flu_dpm:.0f}, bad_flu_dpm: {bad_flu_dpm:.0f}')


def days_since_trim(df, index_col, days_since_col, days_since_thresh, values_col=None):
    '''
    The visualizations look bad when some of the entities have very long trajectories (e.g. China) and others have
    much shorter ones.

    Trim the data for visualizations s.t. data before the date the threshold is reached is removed.
    If the index_col is 'days_since', create the days_since column.

    :param df:
    :param index_col: either date or days_since. This is the pivot index.
    :param days_since_col: e.g. deaths_per_million or date or cases_per_day
    :param days_since_thresh: e.g. 0.1 deaths per million or pd.to_datetime('2020-03-01') or 1 case per day
    :param values_col: e.g. deaths, deaths_per_million, deaths_per_day, deaths_per_million_per_day. If None,
    the default, the entire row of observations before the first day the threshold is met is removed. If it
    is the name of a column, that column is filled with np.nan for those rows.
    :return:
    '''
    # Trim rows before the timeperiod we are looking at:
    df = df.copy()
    if days_since_col:
        df[values_col] = fill_before_first(df, values_col, days_since_thresh, thresh_col=days_since_col)
        if index_col == 'days_since':
            df['days_since'] = make_days_since(df, days_since_col, days_since_thresh)

    return df


def prioritize_entities(df, index_col, values_col, ascending=True, n_top=None, n_show=None, includes=None, excludes=None):
    # Pivot to a table with country/entity columns and date/days_since rows
    piv = df.pivot(index=index_col, columns='entity', values=values_col)
    piv = piv.loc[piv.notnull().any(axis=1), :]  # remove entities with all null values

    # sort by the last non-nan value. (For index_col=='days_since', the last value(s) can be nan.
    sort_idx = np.argsort(piv.apply(lambda s: s[s.notna()].iloc[-1], axis=0))
    if not ascending:
        sort_idx = sort_idx[::-1]

    # Choose which entities to plot
    sorted_entities = piv.columns.values[sort_idx]
    n_ent = len(sorted_entities)
    n_top = n_ent if n_top is None else n_top
    n_show = n_ent if n_show is None else (n_top if n_top > n_show else n_show)
    print(f'n_top: {n_top}, n_show: {n_show}, includes: {includes}, excludes: {excludes}')
    includes_idx = np.isin(sorted_entities, includes) if includes else np.zeros_like(sorted_entities, dtype=bool)
    excludes_idx = np.isin(sorted_entities, excludes) if excludes else np.zeros_like(sorted_entities, dtype=bool)
    priority_idx = np.hstack([np.arange(n_ent)[includes_idx & ~excludes_idx],
                              np.arange(n_ent)[~includes_idx & ~excludes_idx]])
    show_entities = sorted_entities[np.sort(priority_idx[:n_show])]
    top_entities = sorted_entities[np.sort(priority_idx[:n_top])]
    return top_entities, show_entities, sorted_entities


def plot_trajectories(df, index_col='date', values_col='deaths', rank=False, n_top=None,
                      ascending=False,
                      includes=None, excludes=None, n_show=None, log_yaxis=False,
                      show_legend=True, title=None,
                      ):
    '''
    df: columns: date, deaths, cases, population
    index_col: either date or days_since. This is the pivot index.
    values_col: e.g. deaths, deaths_per_million, deaths_per_day, deaths_per_million_per_day
    rank: plot the ranks of the values within each day.
    n_top: int > 0. Display the n highest trajectories on the chart.
    ascending: By default, "top_n" is sorted in descending order to show the worst off entities.
      For `rank=True` or to show entities that are hit least hard, sort in ascending order.
    n_show: int > 0. Show at most n on the chart.
    includes: array of entities to highlight.
    excludes: array of entities to exclude from chart.
    log_yaxis: if True, plot y-axis on a log scale.
    show_legend: add a legend to the plot. Sometimes the legend makes the plot less legible.
    title: if None, a title is autogenerated.
    '''
    log_yaxis = log_yaxis if log_yaxis is not None else (not rank and 'ratio' not in values_col)

    # https://stackoverflow.com/questions/13851535/delete-rows-from-a-pandas-dataframe-based-on-a-conditional-expression-involving
    if excludes:
        df = df.loc[~df['entity'].isin(excludes), :]

    # remove obsevations without values
    df = df.loc[df[values_col].notna(), :]
    # remove observations without index value
    # for example, days that happen before the first day of the days_since column
    df = df.loc[df[index_col].notna(), :]

    # Pivot to a table with country/entity columns and date/days_since rows
    piv = df.pivot(index=index_col, columns='entity', values=values_col)
    piv = piv.loc[piv.notnull().any(axis=1), :]  # remove rows with all null values

    # entities ranked by each day, or by each day since 0.1.
    if rank:
        piv = piv.rank(axis=1, method='average', ascending=False)

    # Plot countries in order, sorting by the most recent value for each entity.
    # For days_since, the last value can be nan. Find the most recent non-nan value.
    sort_idx = np.argsort(piv.apply(lambda s: s[s.notna()].iloc[-1], axis=0))
    if not (rank or ascending):
        # plot deaths per million from largest to smallest
        sort_idx = sort_idx[::-1]
        pass

    # Choose which entitites to plot
    sorted_entities = piv.columns.values[sort_idx]
    #     print('Num sorted_entities:', len(sorted_entities))
    n_ent = len(sorted_entities)
    n_top = n_ent if n_top is None else n_top
    n_show = n_ent if n_show is None else n_show
    includes_idx = np.isin(sorted_entities, includes) if includes else np.zeros_like(sorted_entities, dtype=bool)
    excludes_idx = np.isin(sorted_entities, excludes) if excludes else np.zeros_like(sorted_entities, dtype=bool)
    priority_idx = np.hstack([np.arange(n_ent)[includes_idx & ~excludes_idx],
                              np.arange(n_ent)[~includes_idx & ~excludes_idx]])
    show_entities = sorted_entities[np.sort(priority_idx[:n_show])]
    top_entities = sorted_entities[np.sort(priority_idx[:n_top])]
    # print(len(show_entities), show_entities)
    # Figure
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, entity in enumerate(show_entities):
        if entity in top_entities:
            linewidth = 2.0
            alpha = 1.0
            entity_rank = np.arange(n_ent)[sorted_entities == entity][0] + 1
            label = f'{entity}[{entity_rank}]'
            annotation = entity
            last_idx = piv.index[piv[entity].notna()].values[-1]
        else:
            linewidth = 1.0
            alpha = 0.5
            label = None
            annotation = None

        if log_yaxis:
            plt.semilogy(piv.index, piv[entity], label=label, linewidth=linewidth, alpha=alpha,
                         marker='o', markersize='4')
        else:  # if rank or 'ratio' in values_col or not log_yaxis:
            plt.plot(piv.index, piv[entity], label=label, linewidth=linewidth, alpha=alpha,
                     marker='o', markersize='4')

        if annotation:
            plt.annotate(entity, xy=(last_idx, piv[entity].loc[last_idx]))

    # pivot == 'days_since' maybe does not play well with bad_flu in the data.
    if values_col == 'deaths_per_million' and not rank:
        plt.axhline(seasonal_flu_dpm, color='blue', linestyle='--', label='seasonal flu')
        plt.axhline(bad_flu_dpm, color='orange', linestyle='--', label='bad flu')

    if values_col.endswith('_ratio'):
        # plt.axhline(1, color='blue', linestyle='--', label='same')
        pass

    plt.grid(True, which='major')  # add gridlines (major and minor ticks)

    if rank:
        ax.invert_yaxis()

    ylabel = values_col
    if rank:
        ylabel += ' (rank)'

    if log_yaxis:
        ylabel += ' (log scale)'

    plt.ylabel(ylabel)

    if index_col == 'date':
        plt.xlabel('Date')
    elif index_col == 'days_since':
        plt.xlabel(f'Days since outbreak began')

    if title is None:
        title = f'{values_col}{"" if not rank else " (rank)"} in Each Location'
        title += ' Over Time' if index_col != 'days_since' else f' Since Outbreak Began'

    plt.title(title)

    plt.xticks(rotation=-60)
    if show_legend:
        plt.legend(loc='upper left')

    plt.show()
    return piv


def plot_two_trajectories(df, index_col, col1, col2, entities):
    label1 = col1
    label2 = col2
    color1 = 'blue'
    color2 = 'orange'
    linestyle1 = 'solid'
    linestyle2 = 'dashed'

    def plot_axis(ax, col, label, color, linestyle):
        piv = df.pivot(index=index_col, columns='entity', values=col)
        # Pivot to a table with country/entity columns and date/days_since rows
        for entity in entities:
            alpha = 1.0
            linewidth = 2.0
            ax.plot(piv.index, piv[entity], label=label, linewidth=linewidth, alpha=alpha,
                    marker='o', markersize='4', color=color, linestyle=linestyle)

            last_index_value = piv.index[piv[entity].notna()].values[-1]
            ax.annotate(entity,
                        xy=(last_index_value, piv[entity].loc[last_index_value]),
                        fontsize=20)

        ax.tick_params(axis='y', labelcolor=color, labelsize=14)
        ax.set_ylabel(label, color=color, fontsize=16)

    fig, ax1 = plt.subplots(figsize=(16, 8))
    plot_axis(ax1, col1, label1, color1, linestyle1)
    ax2 = ax1.twinx()
    plot_axis(ax2, col2, label2, color2, linestyle2)
    ax1.set_xlabel('Date', fontsize=16)
    ax1.tick_params(axis='x', labelsize=14, rotation=-60)
    fig.suptitle(f'{col1} and {col2} \n for {", ".join(entities)}', fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, .9))  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_multi_trajectories(df, index_col='date', values_col='deaths', rank=False, n_top=None,
                      ascending=False,
                      includes=None, excludes=None, n_show=None, log_yaxis=False,
                      show_legend=True, title=None,
                      ):
    '''
    df: columns: date, deaths, cases, population
    index_col: either date or days_since. This is the pivot index.
    values_col: e.g. deaths, deaths_per_million, deaths_per_day, deaths_per_million_per_day
    rank: plot the ranks of the values within each day.
    n_top: int > 0. Display the n highest trajectories on the chart.
    ascending: By default, "top_n" is sorted in descending order to show the worst off entities.
      For `rank=True` or to show entities that are hit least hard, sort in ascending order.
    n_show: int > 0. Show at most n on the chart.
    includes: array of entities to highlight.
    excludes: array of entities to exclude from chart.
    log_yaxis: if True, plot y-axis on a log scale.
    show_legend: add a legend to the plot. Sometimes the legend makes the plot less legible.
    title: if None, a title is autogenerated.
    '''
    log_yaxis = log_yaxis if log_yaxis is not None else (not rank and 'ratio' not in values_col)

    # https://stackoverflow.com/questions/13851535/delete-rows-from-a-pandas-dataframe-based-on-a-conditional-expression-involving
    if excludes:
        df = df.loc[~df['entity'].isin(excludes), :]

    # remove obsevations without values
    df = df.loc[df[values_col].notna(), :]
    # remove observations without index value
    # for example, days that happen before the first day of the days_since column
    df = df.loc[df[index_col].notna(), :]

    # Pivot to a table with country/entity columns and date/days_since rows
    piv = df.pivot(index=index_col, columns='entity', values=values_col)
    piv = piv.loc[piv.notnull().any(axis=1), :]  # remove rows with all null values

    # entities ranked by each day, or by each day since 0.1.
    if rank:
        piv = piv.rank(axis=1, method='average', ascending=False)

    # Plot countries in order, sorting by the most recent value for each entity.
    # For days_since, the last value can be nan. Find the most recent non-nan value.
    sort_idx = np.argsort(piv.apply(lambda s: s[s.notna()].iloc[-1], axis=0))
    if not (rank or ascending):
        # plot deaths per million from largest to smallest
        sort_idx = sort_idx[::-1]
        pass

    # Choose which entitites to plot
    sorted_entities = piv.columns.values[sort_idx]
    #     print('Num sorted_entities:', len(sorted_entities))
    n_ent = len(sorted_entities)
    n_top = n_ent if n_top is None else n_top
    n_show = n_ent if n_show is None else n_show
    includes_idx = np.isin(sorted_entities, includes) if includes else np.zeros_like(sorted_entities, dtype=bool)
    excludes_idx = np.isin(sorted_entities, excludes) if excludes else np.zeros_like(sorted_entities, dtype=bool)
    priority_idx = np.hstack([np.arange(n_ent)[includes_idx & ~excludes_idx],
                              np.arange(n_ent)[~includes_idx & ~excludes_idx]])
    show_entities = sorted_entities[np.sort(priority_idx[:n_show])]
    top_entities = sorted_entities[np.sort(priority_idx[:n_top])]
    # print(len(show_entities), show_entities)
    # Figure
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, entity in enumerate(show_entities):
        if entity in top_entities:
            linewidth = 2.0
            alpha = 1.0
            entity_rank = np.arange(n_ent)[sorted_entities == entity][0] + 1
            label = f'{entity}[{entity_rank}]'
            annotation = entity
            last_idx = piv.index[piv[entity].notna()].values[-1]
        else:
            linewidth = 1.0
            alpha = 0.5
            label = None
            annotation = None

        if log_yaxis:
            plt.semilogy(piv.index, piv[entity], label=label, linewidth=linewidth, alpha=alpha,
                         marker='o', markersize='4')
        else:  # if rank or 'ratio' in values_col or not log_yaxis:
            plt.plot(piv.index, piv[entity], label=label, linewidth=linewidth, alpha=alpha,
                     marker='o', markersize='4')

        if annotation:
            plt.annotate(entity, xy=(last_idx, piv[entity].loc[last_idx]))

    # pivot == 'days_since' maybe does not play well with bad_flu in the data.
    if values_col == 'deaths_per_million' and not rank:
        plt.axhline(seasonal_flu_dpm, color='blue', linestyle='--', label='seasonal flu')
        plt.axhline(bad_flu_dpm, color='orange', linestyle='--', label='bad flu')

    if values_col.endswith('_ratio'):
        # plt.axhline(1, color='blue', linestyle='--', label='same')
        pass

    plt.grid(True, which='major')  # add gridlines (major and minor ticks)

    if rank:
        ax.invert_yaxis()

    ylabel = values_col
    if rank:
        ylabel += ' (rank)'

    if log_yaxis:
        ylabel += ' (log scale)'

    plt.ylabel(ylabel)

    if index_col == 'date':
        plt.xlabel('Date')
    elif index_col == 'days_since':
        plt.xlabel(f'Days since outbreak began')

    if title is None:
        title = f'{values_col}{"" if not rank else " (rank)"} in Each Location'
        title += ' Over Time' if index_col != 'days_since' else f' Since Outbreak Began'

    plt.title(title)

    plt.xticks(rotation=-60)
    if show_legend:
        plt.legend(loc='upper left')

    plt.show()
    return piv


def plot_scatter(df, then_col, now_col, since_col, since_thresh, entities=None):
    if entities:
        df = df.loc[df['entity'].isin(entities), :]

    df = df.loc[~before_threshold(df, col=since_col, thresh=since_thresh), :]
    grouped = df.groupby(['entity'])
    ents = list(grouped[then_col].first().index)
    then_values = grouped[then_col].apply(lambda s: s[s.notna()].values[0])
    now_values = grouped[now_col].last()
    now_date = grouped['date'].last().max()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.loglog(then_values, now_values, 'o')
    (xmin, xmax, slope, intercept, p_value, std_err, r_value,
     r_squared, pred_x, pred_y) = fit_power_law(
        then_values, now_values)
    plt.loglog(pred_x, pred_y, ':', label=f"linear model")
    since_thresh_label = since_thresh.strftime("%Y-%m-%d") if isinstance(since_thresh,
                                                                         datetime.datetime) else since_thresh
    ax.set_xlabel(f'{then_col} on the first day that {since_col} >= {since_thresh_label}')
    ax.set_ylabel(f'most recent {now_col} (e.g. {now_date.strftime("%Y-%m-%d")})')
    for ent, then_val, now_val in zip(ents, then_values, now_values):
        ax.annotate(ent, xy=(then_val, now_val))
    ax.set_title(f'{then_col} vs {now_col}\n' +
                 f'slope={slope:.3}, p-value={p_value:.3}, std err={std_err:.3}' +
                 f', R-squared={r_squared:.3}, correlation coef={r_value:.3}')
    plt.show()


def format_for_display(df):
    """
    Round estimates that should be counts. Format frequencies to 6 decimal places to avoid scientific notation when
    printing table.
    :param df: deaths cases tests infections population ifr cfr mr prevalence
    :return:
    """
    # display
    df['infections'] = df['infections'].round()
    df['population'] = df['population'].round()
    df['tests'] = df['tests'].round()
    df['cases'] = df['cases'].round()
    df['deaths'] = df['deaths'].round()
    df['prevalence'] = df['prevalence'].apply(lambda x: f'{x:.6f}')
    df['ifr'] = df['ifr'].apply(lambda x: f'{x:.6f}')
    df['cfr'] = df['cfr'].apply(lambda x: f'{x:.6f}')
    df['mr'] = df['mr'].apply(lambda x: f'{x:.6f}')
    if 'herd_deaths' in df.columns:
        df['herd_deaths'] = df['herd_deaths'].round()

    return df



