import datetime
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import re



DATA_DIR = Path('./data')

# https://www.gov.uk/eu-eea. OWID='Czech Republic', ECDC='Czechia'
eu_countries_owid = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
                     'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland',
                     'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland',
                     'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden']
eu_countries_ecdc = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia',
                     'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland',
                     'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland',
                     'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden']


def download_or_load(url, filename, download=False, cache=False):
    '''
    Download the CSV url and make a dataframe. Or read the csv from a file.
    If downloading, optionally cache the csv to `filename`.
    :param url:
    :param filename:
    :param download: If True, download the url.
    :param cache:  If True, save the downloaded csv.
    :return:
    '''
    if download:
        df = pd.read_csv(url)
        if cache:
            df.to_csv(filename)
    else:
        df = pd.read_csv(filename)

    return df


def make_nday_ratio(df, col, n=1):
    return df.groupby(['entity'])[col].transform(
        lambda s: s.rolling(n + 1).apply(lambda w: w.iloc[-1] / w.iloc[0]))


def make_nday_diff(df, col, n=1):
    return df.groupby(['entity'])[col].transform(
        lambda s: s.rolling(n + 1).apply(lambda w: w.iloc[-1] - w.iloc[0]))


def before_threshold(df, col, thresh, groupby='entity'):
    '''
    Return boolean index of df rows before the first day when the threshold
    was met, i.e. when df[col] >= thresh. This is done with each entity.
    :param df: entity/state/country column, ordered by date column, cases, deaths, etc.
    '''
    return df.groupby(groupby)[col].transform(lambda s: (s >= thresh).cumsum() == 0)


def fill_before_first(df, col, thresh, thresh_col=None, fill=np.nan):
    '''
    For each entity, replace the values of col for every row that occurs
    before the first row where thresh_col >= thresh.
    For example, fill in the 'deaths' column with np.nan for every row before the
    first date where 'deaths_per_million' was >= 0.1.
    :param col: the column to be filled in
    :param thresh: the threshold value used to determine what rows occur before the
      threshold is first met.
    :param thresh_col: if not None, use this column to determine which rows to fill in.
    :param fill: the fill value, np.nan by default.
    '''
    return df[col].where(~before_threshold(df, thresh_col, thresh), fill)


def make_days_since(df, col, thresh):
    '''
    Days since first day the value of col >= thresh.
    Example of column and threshold from data journalists:
    - days since deaths per million reached 0.1 deaths per million
      (OWID deaths per million trajectories).
    - days since 3 daily deaths first recorded (FT daily deaths trajectory)
      https://twitter.com/jburnmurdoch/status/1245466020053164034

    Example usage:
        df['days_since'] = make_days_since(df, 'deaths_per_million', 0.1)
    '''

    # >>> a = np.array([[1,1,1,1],[1,2,1,2],[1,1,2,2],[2,2,2,2]]).astype(float)
    # >>> a
    # array([[1., 1., 1., 1.],
    #        [1., 2., 1., 2.],
    #        [1., 1., 2., 2.],
    #        [2., 2., 2., 2.]])
    # >>> b = ((a > 1).cumsum(axis=1) > 0).cumsum(axis=1).astype(float) - 1
    # >>> b[b < 0] = np.nan
    # >>> b
    # array([[nan, nan, nan, nan],
    #        [nan,  0.,  1.,  2.],
    #        [nan, nan,  0.,  1.],
    #        [ 0.,  1.,  2.,  3.]])
    def days_since(s, thresh):
        '''s: a Series containing the values of col for an entity ordered by date'''
        days = ((s >= thresh).cumsum() > 0).cumsum().astype(float) - 1
        days[days < 0] = np.nan
        return days

    return df.groupby(['entity'])[col].transform(days_since, thresh=thresh)


def add_per_day(df, columns):
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        df[f'{col}_per_day'] = df.groupby('entity')[col].transform(
            lambda s: s.rolling(2).apply(lambda w: w.iloc[-1] - w.iloc[0]))

    return df


def add_nday_avg(df, columns, n=7):
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        df[f'{col}_{n}day_avg'] = df.groupby('entity')[col].transform(
            lambda s: s.rolling(n).mean())

    return df


def add_derived_values_cols(df):
    '''
    deaths_per_million
    cases_per_test
    cases_per_day
    etc., etc., etc.
    '''
    values_cols = ['cases', 'deaths'] + (['tests'] if 'tests' in df.columns else [])
    for values in values_cols:
        values_per_million = values + '_per_million'
        df[values_per_million] = df[values] / df['population'] * 1e6

        values_per_day = values + '_per_day'
        df[values_per_day] = df.groupby(['entity'])[values].transform(
            lambda s: s.rolling(2).apply(lambda w: w.iloc[-1] - w.iloc[0]))

        # FT: 7-day rolling average of daily change in deaths
        # https://www.ft.com/coronavirus-latest
        values_per_day_7day_avg = values_per_day + '_7day_avg'
        df[values_per_day_7day_avg] = (df.groupby(['entity'])[values_per_day]
                                       .transform(lambda s: s.rolling(7).mean()))

        values_per_day_3day_avg = values_per_day + '_3day_avg'
        df[values_per_day_3day_avg] = (df.groupby(['entity'])[values_per_day]
                                       .transform(lambda s: s.rolling(3).mean()))

        values_per_million_per_day = values_per_million + '_per_day'
        df[values_per_million_per_day] = df.groupby(['entity'])[values_per_million].transform(
            lambda s: s.rolling(2).apply(lambda w: w.iloc[-1] - w.iloc[0]))

        # Per capita alternative to unnormalized FT values
        values_per_million_per_day_7day_avg = values_per_million_per_day + '_7day_avg'
        df[values_per_million_per_day_7day_avg] = (df.groupby(['entity'])[values_per_million_per_day]
                                                   .transform(lambda s: s.rolling(7).mean()))

        values_per_million_per_day_3day_avg = values_per_million_per_day + '_3day_avg'
        df[values_per_million_per_day_3day_avg] = (df.groupby(['entity'])[values_per_million_per_day]
                                                   .transform(lambda s: s.rolling(3).mean()))

        values_per_day_3day_avg_14day_ratio = values_per_day_3day_avg + '_14day_ratio'
        df[values_per_day_3day_avg_14day_ratio] = df.groupby(['entity'])[values_per_day_3day_avg].transform(
            lambda s: s.rolling(15).apply(lambda w: w.iloc[-1] / w.iloc[0]))

    # Case Fatality Rate
    df['deaths_per_case'] = df['deaths'] / df['cases']
    df['deaths_per_case_per_day_7day_avg'] = df['deaths_per_day_7day_avg'] / df['cases_per_day_7day_avg']
    df['deaths_per_case_per_day_3day_avg'] = df['deaths_per_day_3day_avg'] / df['cases_per_day_3day_avg']

    if 'tests' in df.columns:
        for values in ['cases', 'deaths']:
            # Various forms of Cases / Test
            df[f'{values}_per_test'] = df[values] / df['tests']
            df[f'{values}_per_day_3day_avg_tests_per_day_3day_avg_ratio'] = (
                    df[f'{values}_per_day_3day_avg'] / df['tests_per_day_3day_avg'])
            df[f'{values}_per_day_7day_avg_tests_per_day_7day_avg_ratio'] = (
                    df[f'{values}_per_day_7day_avg'] / df['tests_per_day_7day_avg'])
            df[f'{values}_per_test_per_day'] = df[f'{values}_per_day'] / df['tests_per_day']
            df[f'{values}_per_test_per_day_3day_avg'] = (df.groupby('entity')[f'{values}_per_test_per_day']
                                                         .transform(lambda s: s.rolling(3).mean()))
            df[f'{values}_per_test_per_day_7day_avg'] = (df.groupby('entity')[f'{values}_per_test_per_day']
                                                         .transform(lambda s: s.rolling(7).mean()))
            df[f'{values}_per_day_3day_avg_tests_per_day_3day_avg_ratio_14day_ratio'] = (
                df.groupby(['entity'])[f'{values}_per_day_3day_avg_tests_per_day_3day_avg_ratio']
                    .transform(lambda s: s.rolling(15).apply(lambda w: w.iloc[-1] / w.iloc[0]))
            )
            df[f'{values}_per_test_per_day_3day_avg_14day_ratio'] = (
                df.groupby(['entity'])[f'{values}_per_test_per_day_3day_avg']
                    .transform(lambda s: s.rolling(15).apply(lambda w: w.iloc[-1] / w.iloc[0]))
            )
            df[f'{values}_per_day_7day_avg_tests_per_day_7day_avg_ratio_14day_ratio'] = (
                df.groupby(['entity'])[f'{values}_per_day_7day_avg_tests_per_day_7day_avg_ratio']
                    .transform(lambda s: s.rolling(15).apply(lambda w: w.iloc[-1] / w.iloc[0]))
            )
            df[f'{values}_per_test_per_day_7day_avg_14day_ratio'] = (
                df.groupby(['entity'])[f'{values}_per_test_per_day_7day_avg']
                    .transform(lambda s: s.rolling(15).apply(lambda w: w.iloc[-1] / w.iloc[0]))
            )

    return df


def seroprevalence_adjustments(df):
    '''
    :param df: entity, population
    :return:
    '''
    # seroprevalence adjustment factors
    # ny data from 2020-05-02
    ny_seroprevalence = 0.123
    ny_tests = 959071
    ny_cases = 312977
    ny_pop = df.loc[df['entity'] == 'New York', 'population'].values[0]
    # print('ny_pop', ny_pop)
    erickson_adjustment = ny_seroprevalence / (ny_cases / ny_tests)
    confirmed_adjustment = ny_seroprevalence / (ny_cases / ny_pop)
    return erickson_adjustment, confirmed_adjustment


def to_prevalence_dataframe(df):
    '''
    Consider methods of estimating prevalence and infection fatality rate.
    Add a new dimension/axis to df by repeating each row three times, once for
    each method and adding a 'method' and 'prevalence' and 'ifr' columns.

    Methods:
    - confirmed: cases/population
    - erickson: cases/tests
    - adjusted_erickson: using a seroprevalence adjusted cases/tests.
    - adjusted_confirmed: using a seroprevalence adjusted cases/population

    :param df: date, entity, deaths, cases, tests, population.
    :return: a dataframe with a new "dimension" column, method, and 2 new value columns, prevalence and ifr.
    '''

    # Add 'method' dimension to df
    # entity, date, method, deaths, cases, tests, population
    methods = ['confirmed', 'adjusted_confirmed', 'erickson', 'adjusted_erickson']
    entities = df['entity'].unique()
    ents, mets = list(zip(*itertools.product(entities, methods)))
    dm = pd.DataFrame({'entity': ents, 'method': mets})
    dg = df.merge(dm, on='entity')  # add "method" dimension

    erickson_adjustment, confirmed_adjustment = seroprevalence_adjustments(dg)
    idx = dg['method'] == 'confirmed'
    dg.loc[idx, 'prevalence'] = dg.loc[idx, 'cases'] / dg.loc[idx, 'population']
    idx = dg['method'] == 'adjusted_confirmed'
    dg.loc[idx, 'prevalence'] = confirmed_adjustment * dg.loc[idx, 'cases'] / dg.loc[idx, 'population']
    idx = dg['method'] == 'erickson'
    dg.loc[idx, 'prevalence'] = dg.loc[idx, 'cases'] / dg.loc[idx, 'tests']
    idx = dg['method'] == 'adjusted_erickson'
    dg.loc[idx, 'prevalence'] = erickson_adjustment * dg.loc[idx, 'cases'] / dg.loc[idx, 'tests']
    dg.loc[:, 'ifr'] = dg.loc[:, 'deaths'] / (dg.loc[:, 'prevalence'] * dg.loc[:, 'population'])
    return dg


def to_prevalence_age_dataframe(df, pop, dc):
    '''
    We have data for deaths and cases by state and age band
    We have data for population by state and age band
    We have data for deaths and cases by state and date.
    We would like to have: tests by state and age band, and by date.
    We would love to have: comorbidities by state and age band, and by date.
    :param df: entity date deaths cases tests.
    :param pop: entity age_band population
    :param dc: entity age_band deaths and cases. Only for a single entity. To be broadcast across entities.
    :return:
    '''
    dc = dc[dc['entity'] == dc['entity'].iloc[0]]  # insist on only a single entity in dc.
    # add 'method' and 'age_band' dimensions to df
    # entity, date, age_band, method, deaths, cases, tests, population, population_age, cases_age, deaths_age
    methods = ['confirmed', 'adjusted_confirmed', 'erickson', 'adjusted_erickson']
    entities = df['entity'].unique()
    ents, mets = list(zip(*itertools.product(entities, methods)))
    dm = pd.DataFrame({'entity': ents, 'method': mets})
    # _age columns are broadcast across date dimension
    # non _age columns are broadcast across age dimension
    # all value columns are broadcast across method dimension
    dg = df.merge(pop, on='entity', suffixes=('', '_age'))  # broadcast population_age across dates, methods
    dg = dg.merge(dc, on=['age_band'], suffixes=('', '_age'))  # broadcast deaths_age, cases_age across entities, methods, dates.
    dg = dg.merge(dm, on='entity')  # add "method" dimension

    age_bands = dg['age_band'].unique()
    n_age = len(age_bands)

    # print(f'dg.columns: {dg.columns}')
    # print(dg[dg['entity'] == 'New York'])
    erickson_adjustment, confirmed_adjustment = seroprevalence_adjustments(dg)
    tests_age = dg['tests'] / n_age  # we don't have tests by age and state.
    population_age = dg['population_age']

    dg['cases_age_frac'] = dg['cases_age'] / dg.groupby(['entity', 'date', 'method'])['cases_age'].transform(sum)
    cases_age = dg['cases_age_frac'] * dg['cases']  # methods: uses cases * the proportion of cases from the age band
    # cases_age = dg['cases'] / n_age  # methods: distribute cases uniformly across age band.  instead of cases / n.

    dg['deaths_age_frac'] = dg['deaths_age'] / dg.groupby(['entity', 'date', 'method'])['deaths_age'].transform(sum)
    deaths_age = dg['deaths_age_frac'] * dg['deaths']  # methods: uses cases * the proportion of cases from the age band
    # deaths_age = dg['deaths'] / n_age  # methods: uses cases * pct_cases_age instead of cases / n.

    idx = dg['method'] == 'confirmed'
    dg.loc[idx, 'prevalence'] = cases_age[idx] / population_age[idx]
    idx = dg['method'] == 'adjusted_confirmed'
    dg.loc[idx, 'prevalence'] = (confirmed_adjustment * cases_age[idx]) / population_age[idx]
    idx = dg['method'] == 'erickson'
    dg.loc[idx, 'prevalence'] = cases_age[idx] / tests_age[idx]
    idx = dg['method'] == 'adjusted_erickson'
    dg.loc[idx, 'prevalence'] = (erickson_adjustment * cases_age[idx]) / tests_age[idx]

    dg.loc[:, 'ifr'] = deaths_age / (dg.loc[:, 'prevalence'] * population_age)
    return dg


def add_prevalence_dimension(df):
    entities = df['entity'].unique()
    methods = ['confirmed', 'adjusted_confirmed', 'erickson', 'adjusted_erickson']
    ents, mets = list(zip(*itertools.product(entities, methods)))
    dm = pd.DataFrame({'entity': ents, 'method': mets})
    dg = df.merge(dm, on='entity')
    return dg


def add_prevalence_cols(df, groupby=['entity']):
    """

    :param df: method [entity] [age_band] [herd] [comorbidity] tests cases deaths population
    :return:
    """
    erickson_adjustment, confirmed_adjustment = seroprevalence_adjustments(df)
    idx = df['method'] == 'confirmed'
    df.loc[idx, 'prevalence'] = df.loc[idx, 'cases'] / df.loc[idx, 'population']
    idx = df['method'] == 'adjusted_confirmed'
    df.loc[idx, 'prevalence'] = confirmed_adjustment * df.loc[idx, 'cases'] / df.loc[idx, 'population']
    idx = df['method'] == 'erickson'
    df.loc[idx, 'prevalence'] = df.loc[idx, 'cases'] / df.loc[idx, 'tests']
    idx = df['method'] == 'adjusted_erickson'
    df.loc[idx, 'prevalence'] = erickson_adjustment * df.loc[idx, 'cases'] / df.loc[idx, 'tests']
    df.loc[:, 'ifr'] = df.loc[:, 'deaths'] / (df.loc[:, 'prevalence'] * df.loc[:, 'population'])
    return df


def add_age_dimension(df, pop, dc):
    '''
    We have data for deaths and cases by state and age band
    We have data for population by state and age band
    We have data for deaths and cases by state and date.
    We would like to have: tests by state and age band, and by date.
    We would love to have: comorbidities by state and age band, and by date.
    :param df: entity date deaths cases tests.
    :param pop: entity age_band population
    :param dc: entity age_band deaths cases. Only for a single entity. To be broadcast across entities.
    :return:
    '''
    dc = dc[dc['entity'] == dc['entity'].iloc[0]]  # insist on only a single entity in dc.
    dg = df.merge(pop, on='entity', suffixes=('', '_age'))  # broadcast population_age across dates, methods
    dg = dg.merge(dc, on='age_band', suffixes=('', '_age'))  # broadcast deaths_age, cases_age across entities, methods, dates.
    return dg


def add_prevalence_age_cols(dg, groupby=['entity', 'date', 'method']):
    """

    :param dg: entity date method age_band tests cases deaths population
    :return:
    """
    # print(f'dg.columns: {dg.columns}')
    # print(dg[dg['entity'] == 'New York'])
    erickson_adjustment, confirmed_adjustment = seroprevalence_adjustments(dg)
    n_age = len(dg['age_band'].unique())
    tests_age = dg['tests'] / n_age  # we don't have tests by age and state.
    population_age = dg['population_age']

    dg['cases_age_frac'] = dg['cases_age'] / dg.groupby(groupby)['cases_age'].transform(sum)
    cases_age = dg['cases_age_frac'] * dg['cases']  # methods: uses cases * the proportion of cases from the age band
    # cases_age = dg['cases'] / n_age  # methods: distribute cases uniformly across age band.  instead of cases / n.

    dg['deaths_age_frac'] = dg['deaths_age'] / dg.groupby(groupby)['deaths_age'].transform(sum)
    deaths_age = dg['deaths_age_frac'] * dg['deaths']  # methods: uses cases * the proportion of cases from the age band
    # deaths_age = dg['deaths'] / n_age  # methods: uses cases * pct_cases_age instead of cases / n.

    idx = dg['method'] == 'confirmed'
    dg.loc[idx, 'prevalence'] = cases_age[idx] / population_age[idx]
    idx = dg['method'] == 'adjusted_confirmed'
    dg.loc[idx, 'prevalence'] = (confirmed_adjustment * cases_age[idx]) / population_age[idx]
    idx = dg['method'] == 'erickson'
    dg.loc[idx, 'prevalence'] = cases_age[idx] / tests_age[idx]
    idx = dg['method'] == 'adjusted_erickson'
    dg.loc[idx, 'prevalence'] = (erickson_adjustment * cases_age[idx]) / tests_age[idx]

    dg.loc[:, 'ifr'] = deaths_age / (dg.loc[:, 'prevalence'] * population_age)
    return dg


def add_herd_dimension(df):
    '''

    :param df: entity date method deaths cases tests population
    :return:
    '''
    entities = df['entity'].unique()
    herds = [0.4, 0.6, 0.8]
    ents, hers = list(zip(*itertools.product(entities, herds)))
    dm = pd.DataFrame({'entity': ents, 'herd': hers})
    dg = df.merge(dm, on=['entity'])
    return dg


def add_herd_cols(dg, use_pop_age=False):
    population = 'population_age' if use_pop_age else 'population'
    dg['herd_cases'] = dg['herd'] * dg[population]
    dg['herd_deaths'] = dg['herd_cases'] * dg['ifr']
    dg['herd_deaths_per_million'] = 1000000 * dg['herd_deaths'] / dg[population]
    return dg


def to_nyc_band(df):
    '''
    df is a census population dataframe with 'age_band' and 'population' columns. Assign those
    population numbers to the New York City age bands.
    This is can be used with df.groupby().apply() to convert census age group population data.
    :param df:
    :return: a dataframe with 'age_band' and 'population' columns.
    '''
    # Methods: census bands and nyc bands don't intersect perfectly. There is
    # overlap between boundaries at nyc boundaries (0-17, 18-44) and census boundaries (15-19, 20-24).
    # here it is put into the 0 to 17 category, since 3 of 5 years are in that category.
    # Population allocation can be done better by
    # allocating 3/5ths to one category and 2/5ths to the other.
    age_bands = ['0 to 17', '18 to 44', '45 to 64', '65 to 74', '75+']
    pop_0_to_17 = (df.loc[(df['AGEGRP'] >= 1) & (df['AGEGRP'] <= 3), 'population'].sum()
                   + round(df.loc[(df['AGEGRP'] == 4), 'population'].values[0] * 3 / 5))
    pop_18_to_44 = (round(df.loc[(df['AGEGRP'] == 4), 'population'].values[0] * 2 / 5)
                    + df.loc[(df['AGEGRP'] >= 5) & (df['AGEGRP'] <= 9), 'population'].sum())
    pop_45_to_64 = df.loc[(df['AGEGRP'] >= 10) & (df['AGEGRP'] <= 13), 'population'].sum()
    pop_65_to_74 = df.loc[(df['AGEGRP'] >= 14) & (df['AGEGRP'] <= 15), 'population'].sum()
    pop_75_plus = df.loc[(df['AGEGRP'] >= 16) & (df['AGEGRP'] <= 18), 'population'].sum()
    return pd.DataFrame({
        'age_band': age_bands,
        'population': [pop_0_to_17, pop_18_to_44, pop_45_to_64, pop_65_to_74, pop_75_plus]
    })


def to_ma_band(df):
    '''
    Massachusetts reports cases and deaths for the age bands:
    0-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80+
    :param df:
    :return:
    '''
    age_bands = ['0 to 19', '20 to 29', '30 to 39', '40 to 49', '50 to 59', '60 to 69', '70 to 79', '80+']
    pops = [
        df.loc[(df['AGEGRP'] >= 1) & (df['AGEGRP'] <= 4), 'population'].sum(),
        df.loc[(df['AGEGRP'] >= 5) & (df['AGEGRP'] <= 6), 'population'].sum(),
        df.loc[(df['AGEGRP'] >= 7) & (df['AGEGRP'] <= 8), 'population'].sum(),
        df.loc[(df['AGEGRP'] >= 9) & (df['AGEGRP'] <= 10), 'population'].sum(),
        df.loc[(df['AGEGRP'] >= 11) & (df['AGEGRP'] <= 12), 'population'].sum(),
        df.loc[(df['AGEGRP'] >= 13) & (df['AGEGRP'] <= 14), 'population'].sum(),
        df.loc[(df['AGEGRP'] >= 15) & (df['AGEGRP'] <= 16), 'population'].sum(),
        df.loc[(df['AGEGRP'] >= 17) & (df['AGEGRP'] <= 18), 'population'].sum(),
    ]
    return pd.DataFrame({
        'age_band': age_bands,
        'population': pops
    })

def to_census_band(df):
    age_bands = ['0 to 4', '5 to 9', '10 to 14', '15 to 19', '20 to 24', '25 to 29', 
                 '30 to 34', '35 to 39', '40 to 44', '45 to 49', '50 to 54', '55 to 59', 
                 '60 to 64', '65 to 69', '70 to 74', '75 to 79', '80 to 84', '85+']
    pops = [df.loc[df['AGEGRP'] == i, 'population'].sum() for i in range(1, 19)]
    return pd.DataFrame({
        'age_band': age_bands,
        'population': pops
    })


def load_census_state_population_by_age_data(bands='census'):
    '''
    See data/cc-est2018-alldata.pdf for a description of column values for SUMLEV, YEAR and age_band.
    Census groups are 5 years wide. They are mapped to NYC age bands in a crude way.
    bands: 'census' or 'nyc' or 'ma'
    '''
    AGEGRP_to_name = {
        0: 'Total',
        1: '0 to 4',
        2: '5 to 9',
        3: '10 to 14',
        4: '15 to 19',
        5: '20 to 24',
        6: '25 to 29',
        7: '30 to 34',
        8: '35 to 39',
        9: '40 to 44',
        10: '45 to 49',
        11: '50 to 54',
        12: '55 to 59',
        13: '60 to 64',
        14: '65 to 69',
        15: '70 to 74',
        16: '75 to 79',
        17: '80 to 84',
        18: '85+'
    }
    df = pd.read_csv(DATA_DIR / 'cc-est2018-alldata.csv', encoding='latin-1')
    df = df.loc[(df['SUMLEV'] == 50) & (df['YEAR'] == 11),  # county-level, 2018 estimate
                ['STATE', 'STNAME', 'AGEGRP', 'TOT_POP', 'COUNTY', 'CTYNAME']]
    df = (df
          .groupby(['STNAME', 'AGEGRP'])['TOT_POP'].sum().reset_index()
          .rename(columns={'STNAME': 'entity', 'TOT_POP': 'population'})
          )

    if bands == 'nyc':
        df = df.groupby(['entity'])[['AGEGRP', 'population']].apply(to_nyc_band).reset_index().drop(columns='level_1')
    elif bands == 'ma':
        df = df.groupby(['entity'])[['AGEGRP', 'population']].apply(to_ma_band).reset_index().drop(columns='level_1')
    elif bands == 'census':
        df = df.groupby(['entity'])[['AGEGRP', 'population']].apply(to_census_band).reset_index().drop(columns='level_1')
    else:
        raise Exception('Unrecognized bands.', bands)

    return df


def load_un_country_population_data():
    # Adding country population data...
    # map UN country names to OWID country names
    u2o = dict([
        ('Bolivia (Plurinational State of)', 'Bolivia'),
        ('Brunei Darussalam', 'Brunei'),
        ('Cabo Verde', 'Cape Verde'),
        ("Côte d'Ivoire", "Cote d'Ivoire"),
        ('Curaçao', 'Curacao'),
        ('Czechia', 'Czech Republic'),
        ('Democratic Republic of the Congo', 'Democratic Republic of Congo'),
        ('Faroe Islands', 'Faeroe Islands'),
        ('Falkland Islands (Malvinas)', 'Falkland Islands'),
        ('Iran (Islamic Republic of)', 'Iran'),
        ("Lao People's Democratic Republic", 'Laos'),
        ('North Macedonia', 'Macedonia'),
        ('Northern Mariana Islands', 'Mariana Islands'),
        ('Republic of Moldova', 'Moldova'),
        ('State of Palestine', 'Palestine'),
        ('Russian Federation', 'Russia'),
        ('Republic of Korea', 'South Korea'),
        ('Eswatini', 'Swaziland'),
        ('Syrian Arab Republic', 'Syria'),
        ('China, Taiwan Province of China', 'Taiwan'),
        ('United Republic of Tanzania', 'Tanzania'),
        ('Timor-Leste', 'Timor'),
        ('United States of America', 'United States'),
        ('Holy See', 'Vatican'),
        ('Venezuela (Bolivarian Republic of)', 'Venezuela'),
        ('Viet Nam', 'Vietnam')
    ])
    # Load UN population data
    pop_df = pd.read_csv(DATA_DIR / 'WPP2019_TotalPopulationBySex.csv')
    # Use the medium variant projection for the year 2020 of total population
    pop_df = pop_df.loc[(pop_df['Time'] == 2020) & (pop_df['Variant'] == 'Medium'), ['Location', 'PopTotal']]
    # Convert Locations to match OWID country names
    pop_df['Location'] = pop_df['Location'].apply(lambda l: l if l not in u2o else u2o[l])
    pop_df['PopTotal'] = pop_df['PopTotal'] * 1000
    pop_df = pop_df.rename(columns={'PopTotal': 'population'})
    return pop_df


def load_ecdc_country_data(download=False, cache=False):
    """
    Download cases and deaths for countries around the world from the ECDC
    Source: https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide
    Returns: a dataframe with columns date, entity, cases, deaths, population
    """
    url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
    filename = DATA_DIR / 'ecdc_covid19_casedistribution.csv'
    df = download_or_load(url, filename, download=download, cache=cache)
    # fix backwards euro dates
    df['date'] = df.apply(lambda s: datetime.datetime(year=s['year'], month=s['month'], day=s['day']), axis=1)
    df = df.rename(columns={'popData2018': 'population',
                            'countriesAndTerritories': 'entity',
                            'cases': 'new_cases',
                            'deaths': 'new_deaths'})
    df['entity'] = df['entity'].map(lambda e: e.replace('_', ' '))
    fixes = {'United States of America': 'United States'}
    df['entity'] = df['entity'].map(lambda e: e if e not in fixes else fixes[e])
    df = df.sort_values(by=['entity', 'date']).reset_index(drop=True)
    df['cases'] = df.groupby('entity')['new_cases'].cumsum()
    df['deaths'] = df.groupby('entity')['new_deaths'].cumsum()
    df = df.loc[:, ['date', 'entity', 'cases', 'deaths', 'population', 'new_cases', 'new_deaths']]

    # Add Eurpean Union as an entity
    # sanity check: all EU countries are in country population data
    assert len(set(eu_countries_ecdc) - set(df['entity'].unique())) == 0
    eu_pop = df[df['entity'].isin(eu_countries_ecdc)].groupby('entity')['population'].first().sum()
    # aggregate deaths, cases, new_cases, new_deaths for eu countries
    eu_df = (df.loc[df['entity'].isin(eu_countries_ecdc)].groupby('date').aggregate(np.sum)
             .reset_index().assign(entity='EU'))
    eu_df['population'] = eu_pop
    df = df.append(eu_df, ignore_index=True)
    return df


def load_owid_country_data(download=False, cache=False):
    # country covid data
    url = 'https://covid.ourworldindata.org/data/ecdc/full_data.csv'
    filename = DATA_DIR / 'full_data.csv'  # cached version
    df = download_or_load(url, filename, download=download, cache=cache)

    df = df.rename(columns={'total_cases': 'cases',
                            'total_deaths': 'deaths',
                            'location': 'entity'})
    df['date'] = pd.to_datetime(df['date'])
    pop_df = load_un_country_population_data()
    df = (df.merge(pop_df, how='inner', left_on='entity', right_on='Location')
          .drop(columns='Location').rename(columns={'PopTotal': 'population'}))

    ## Add Eurpean Union as an entity
    # sanity check: all EU countries are in country population data
    assert len(eu_countries_owid) == len(pop_df.loc[pop_df['Location'].isin(eu_countries_owid), 'Location'])
    eu_pop = pop_df.loc[pop_df['Location'].isin(eu_countries_owid), 'population'].sum()
    # aggregate deaths, cases, new_cases, new_deaths for eu countries
    eu_df = (df.loc[df['entity'].isin(eu_countries_owid)].groupby('date').aggregate(np.sum)
             .reset_index().assign(entity='EU'))
    eu_df['population'] = eu_pop
    df = df.append(eu_df, ignore_index=True)
    return df


def load_nytimes_us_state_data(download=False, cache=False):
    '''
    nytimes dataset with cases and deaths (cummulative) for US states.
    '''
    url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
    filename = DATA_DIR / 'nytimes-us-states.csv'
    df = download_or_load(url, filename, download=download, cache=cache)

    df = df.rename(columns={'state': 'entity'})
    df['date'] = pd.to_datetime(df['date'])

    # Add state population, used to get deaths per million
    pop_df = (pd.read_csv(DATA_DIR / 'nst-est2019-alldata.csv')
              .loc[lambda d: d['SUMLEV'] == 40,]
              .rename(columns={'POPESTIMATE2019', 'population'})
              )
    df = (df.merge(pop_df, how='inner', left_on='entity', right_on='NAME')
              .loc[:, ['date', 'entity', 'cases', 'deaths', 'population']]
              )

    # Add daily change
    grouped = df.groupby(['entity'])
    df['new_cases'] = grouped['cases'].transform(
        lambda s: s.rolling(2).apply(lambda w: w.iloc[1] - w.iloc[0]))
    df['new_deaths'] = grouped['deaths'].transform(
        lambda s: s.rolling(2).apply(lambda w: w.iloc[1] - w.iloc[0]))
    df['cases_daily_growth_rate'] = grouped['cases'].transform(
        lambda s: s.rolling(2).apply(lambda w: w.iloc[1] / w.iloc[0]))
    df['deaths_daily_growth_rate'] = grouped['deaths'].transform(
        lambda s: s.rolling(2).apply(lambda w: w.iloc[1] / w.iloc[0]))

    return df


def load_usafacts_county_data(download=False, cache=False):
    '''
    Source: https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/
    '''
    cases_url = 'https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv'
    cases_filename = DATA_DIR / 'covid_confirmed_usafacts.csv'
    deaths_url = 'https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_deaths_usafacts.csv'
    deaths_filename = DATA_DIR / 'covid_deaths_usafacts.csv'
    cases = download_or_load(cases_url, cases_filename, download=download, cache=cache)
    deaths = download_or_load(deaths_url, deaths_filename, download=download, cache=cache)
    return cases, deaths


def load_covidtracking_state_data(download=False, cache=False):
    '''
    Source: https://covidtracking.com
    '''
    url = 'https://covidtracking.com/api/v1/states/daily.csv'
    filename = DATA_DIR / 'covidtracking_states_daily.csv'
    df = download_or_load(url, filename, download=download, cache=cache)
    # fix dates
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    # add entity names
    state_abbr = (pd.read_csv(DATA_DIR / 'us_states_abbreviations.csv')
                      .rename(columns={'NAME': 'entity'})
                      .loc[:, ['entity', 'ABBREVIATION']]
                      )
    df = (df.merge(state_abbr, left_on='state', right_on='ABBREVIATION')
          )
    # add population
    state_pop = (pd.read_csv(DATA_DIR / 'nst-est2019-alldata.csv')
                 .loc[lambda d: d['SUMLEV'] == 40, ['NAME', 'POPESTIMATE2019']]  # states
                 .rename(columns={'POPESTIMATE2019': 'population'})
                 )
    df = df.merge(state_pop, how='inner', left_on='entity', right_on='NAME')
    # filter columns and order by date
    df = (df.rename(columns={'positive': 'cases', 'total': 'tests', 'death': 'deaths'})
              .loc[:, ['date', 'entity', 'cases', 'deaths', 'tests', 'population']]
              )
    df = df.sort_values(by=['entity', 'date'])
    return df


# load data
# df['days_since'] = make_days_since(df, 'deaths_per_million', 0.1)
def load_all(download=False):
    states_df = (load_covidtracking_state_data(download=download, cache=True)
                 .pipe(add_derived_values_cols))
    countries_df = (load_ecdc_country_data(download=download, cache=True)
                    .pipe(add_derived_values_cols))
    # combine countries and states, dropping the countries 'Puerto Rico' (b/c it is in states_df) and 'Georgia' (b/c a state is named 'Georgia')
    all_df = (countries_df.loc[
                  ~countries_df['entity'].isin(['Georgia', 'Puerto Rico']), ['date', 'entity', 'cases', 'deaths',
                                                                             'population']]
              .append(states_df.loc[:, ['date', 'entity', 'cases', 'deaths', 'population']], ignore_index=True)
              .reset_index(drop=True)
              .pipe(add_derived_values_cols))
    return states_df, countries_df, all_df


def map_owid_to_un_pop(mer=None):
    '''
    OWID country names and UN country names do not always agree.
    Print where owid countries and un countries do not line up.
    This output can be visually inspected to build a mapping so that all OWID countries map to UN data.
    '''
    # convert UN country names to OWID country names.
    # 'Guernsey': None,  # No population info in UN csv.
    # 'International': None,  # Not a country. :-)
    # 'Jersey': None, # Guernsey and Jersey are british channel islands
    # 'Kosovo': None, # Republic of Kosovo
    u2o = dict([
        ('Bolivia (Plurinational State of)', 'Bolivia'),
        ('Brunei Darussalam', 'Brunei'),
        ('Cabo Verde', 'Cape Verde'),
        ("Côte d'Ivoire", "Cote d'Ivoire"),
        ('Curaçao', 'Curacao'),
        ('Czechia', 'Czech Republic'),
        ('Democratic Republic of the Congo', 'Democratic Republic of Congo'),
        ('Faroe Islands', 'Faeroe Islands'),
        ('Falkland Islands (Malvinas)', 'Falkland Islands'),
        ('Iran (Islamic Republic of)', 'Iran'),
        ("Lao People's Democratic Republic", 'Laos'),
        ('North Macedonia', 'Macedonia'),
        ('Northern Mariana Islands', 'Mariana Islands'),
        ('Republic of Moldova', 'Moldova'),
        ('State of Palestine', 'Palestine'),
        ('Russian Federation', 'Russia'),
        ('Republic of Korea', 'South Korea'),
        ('Eswatini', 'Swaziland'),
        ('Syrian Arab Republic', 'Syria'),
        ('China, Taiwan Province of China', 'Taiwan'),
        ('United Republic of Tanzania', 'Tanzania'),
        ('Timor-Leste', 'Timor'),
        ('United States of America', 'United States'),
        ('Holy See', 'Vatican'),
        ('Venezuela (Bolivarian Republic of)', 'Venezuela'),
        ('Viet Nam', 'Vietnam')
    ])

    if mer is None:
        pop_df = pd.read_csv(DATA_DIR / 'WPP2019_TotalPopulationBySex.csv')
        # Convert Locations to match OWID country names
        pop_df['Location'] = pop_df['Location'].apply(lambda l: l if l not in u2o else u2o[l])
        # Use the medium variant projection for the year 2020 of total population
        pop_df = pop_df.loc[(pop_df['Time'] == 2020) & (pop_df['Variant'] == 'Medium'), ['Location', 'PopTotal']]
        df = load_owid_country_data(download=True, cache=False)
        mer = df.merge(pop_df, how='outer', left_on='entity', right_on='Location')

    print('OWID entity not found in UN pop Location')
    print(mer.loc[mer['Location'].isna(), 'entity'].unique())
    print('UN pop Location not found in OWID entity')
    print(mer.loc[mer['entity'].isna(), 'Location'].unique())
    return mer




def load_ma_state_data():
    return


def load_nyc_age_comorbidity_death_data(cache=False):
    '''
    Return a tidy table containing deaths segmented by age band and comorbidity status.
    Data on Age, Comorbidity and Death
    New York City Sources:
    - NYC Health Department: https://www1.nyc.gov/site/doh/covid/covid-19-data.page
    - deaths by age and comorbidity status: https://www1.nyc.gov/assets/doh/downloads/pdf/imm/covid-19-daily-data-summary-deaths-04222020-1.pdf
    - cases by age band: https://www1.nyc.gov/assets/doh/downloads/pdf/imm/covid-19-daily-data-summary-04222020-1.pdf
    '''
    age_bands = ['0 to 17', '18 to 44', '45 to 64', '65 to 74', '75+', 'unknown']
    min_age = [0, 18, 45, 65, 75, np.nan]
    max_age = [17, 44, 64, 74, 200, np.nan]
    comorbidity = ['yes', 'no', 'unknown']

    deaths_20200421 = [
        3, 0, 0,
        343, 9, 74,
        1917, 44, 282,
        1785, 4, 680,
        3219, 1, 1581,
        0, 0, 2,
    ]
    deaths_20200426 = [
        5, 0, 0,
        379, 10, 90,
        2204, 50, 366,
        2050, 5, 829,
        3794, 1, 1923,
        0, 0, 2,
    ]
    deaths_20200429 = [
        6, 0, 0,
        401, 14, 92,
        2363, 54, 386,
        2255, 4, 842,
        4140, 1, 2011,
        0, 0, 2,
    ]
    # source: https://www1.nyc.gov/assets/doh/downloads/pdf/imm/covid-19-daily-data-summary-deaths-05022020-1.pdf
    deaths_20200501 = [
        6, 0, 0,
        431, 14, 89,
        2532, 60, 345,
        2470, 5, 785,
        4586, 2, 1829,
        0, 0, 2,
    ]

    cases_20200421 = [2839, 51217, 50694, 17474, 15932, 279]
    cases_20200426 = [3488, 57462, 57556, 19545, 17735, 314]
    cases_20200429 = [3711, 59684, 59715, 20273, 18506, 323]
    # source: https://www1.nyc.gov/assets/doh/downloads/pdf/imm/covid-19-daily-data-summary-05022020-1.pdf
    cases_20200501 = [3897, 61376, 61424, 20813, 19045, 328]

    age_comorbidity_df = pd.DataFrame({'age_band': np.repeat(age_bands, len(comorbidity)),
                              'comorbidity': comorbidity * len(age_bands),
                              'deaths': deaths_20200429})
    df = pd.DataFrame({'age_band': age_bands,
                           'min_age': min_age,
                           'max_age': max_age,
                       'cases': cases_20200429,
                       'deaths': list(age_comorbidity_df.groupby('age_band').sum()['deaths'])})
    df = df.loc[df['age_band'] != 'unknown', :]
    # df['cumulative_deaths'] = df['deaths'].cumsum()
    # df['cumulative_death_pct'] = 100 * df['cumulative_deaths'] / df['deaths'].sum()
    # df['cumulative_cases'] = df['cases'].cumsum()
    # df['cumulative_cases_pct'] = 100 * df['cumulative_cases'] / df['cases'].sum()
    df['deaths_per_case'] = df['deaths'] / df['cases']
    # df['cumulative_deaths_per_case'] = df['cumulative_deaths'] / df['cumulative_cases']
    df['entity'] = 'New York'

    # age_df['max_age'] =
    if cache:
        df.to_csv(DATA_DIR / 'nyc_cases_and_deaths_by_age.csv')
        age_comorbidity_df.to_csv(DATA_DIR / 'nyc_deaths_by_age_and_comorbidity.csv')

    return age_comorbidity_df, df


def load_cdc_deaths_by_age(download=False, cache=False):
    url = 'https://data.cdc.gov/api/views/hc4f-j6nb/rows.csv?accessType=DOWNLOAD'
    filename = DATA_DIR / 'cdc_provisional_coronavirus_deaths.csv'
    df = download_or_load(url, filename, download=download, cache=cache)

    df = (df[(df['State'] == 'United States') & (df['Group'] == 'By age')]
          .rename(columns={'State': 'entity',
                           'End week': 'date',
                           'All COVID-19 Deaths (U07.1)': 'deaths',
                           'Indicator': 'age_band'})
          .loc[lambda d: d['age_band'] != 'All ages', ['date', 'entity', 'deaths', 'age_band']]
         )

    def parse_cdc_age_band(age_band):
        if age_band == '85 years and over':
            return 85, 200  # 200 is pre-singularity bullshit max year.
        elif age_band == 'Under 1 year':
            return 0, 0
        elif age_band == 'All ages':
            return 0, 200
        elif (match := re.search('(\d+)\D(\d+) years', age_band)):
            return int(match.group(1)), int(match.group(2))
        else:
            raise Exception('Unparseable age_band', age_band)

    df['min_age'] = df['age_band'].apply(lambda x: parse_cdc_age_band(x)[0])
    df['max_age'] = df['age_band'].apply(lambda x: parse_cdc_age_band(x)[1])
    df = df.sort_values(by=['min_age', 'max_age'], kind='mergesort')
    df['cumulative_deaths'] = df['deaths'].cumsum()
    df['cumulative_death_pct'] = 100 * df['cumulative_deaths'] / df['deaths'].sum()
    return df


def main():
    print(map_owid_to_un_pop())


if __name__ == '__main__':
    main()
