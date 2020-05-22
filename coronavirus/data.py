import datetime
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
import re



DATA_DIR = Path('./data')

# source https://www.mass.gov/doc/covid-19-dashboard-may-6-2020/download
MA_CONDITION_P = 98.3  # on May 6  P(condition|death)
# source: https://www1.nyc.gov/assets/doh/downloads/pdf/imm/covid-19-daily-data-summary-deaths-05072020-1.pdf
NYC_CONDITION_P = 10675 / (10675 + 86)  # on May 6


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

        # values_per_day_per_million = values_per_day + '_per_million'
        # df[values_per_day_per_million] = df[values_per_day] / df['population'] * 1e6

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

        values_per_day_per_million = values_per_day + '_per_million'
        df[values_per_day_per_million] = df[values_per_day] / df['population'] * 1e6

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


def seroprevalence_adjustments():
    '''
    :param df: entity, population
    :return:
    '''
    # seroprevalence adjustment factors
    # ny data from 2020-05-02
    ny_seroprevalence = 0.123
    ny_tests = 959071
    ny_cases = 312977
    ny_pop = 19453561  # from census 2018 population estimate
    erickson_ny = ny_seroprevalence / (ny_cases / ny_tests)
    confirmed_ny = ny_seroprevalence / (ny_cases / ny_pop)
    # https: // www.miamiherald.com / news / coronavirus / article242260406.html
    confirmed_miami = 165000 / 10600  # infections / cases
    # Arizona. Serology data from Ariz. Dept. Health Dashboard.
    az_seroprevalence = 0.028
    az_cases = 11119
    az_pop = 7278717
    confirmed_az = (az_seroprevalence * az_pop) / az_cases
    print('seroprevalence betas: ', erickson_ny, confirmed_ny, confirmed_miami, confirmed_az)
    return erickson_ny, confirmed_ny, confirmed_miami, confirmed_az


def load_seroprevalence():
    '''
    Return prevalence point estimates from antibody serological surveys.
    :return:
    '''
    cols = ['entity', 'date', 'prevalence', 'population', 'infections', 'cases']
    data = [
        ['Arizona', pd.to_datetime('2020-05-10'), 0.028, 7278717, 0.028 * 7278717, 11119],
        ['New York', pd.to_datetime('2020-05-02'), 0.123, 19453561, 0.123 * 19453561, 312977],
        ['Miami-Dade County', pd.to_datetime('2020-04-24'), 0.06, 165000 / 0.06, 165000, 10600],
    ]
    df = pd.DataFrame(data, columns=cols)
    df['factor'] = df['infections'] / df['cases']  # used to estimate prevalence for other entities or dates.
    return df


# methods: default condition_p is an unweighted mean of the nyc and ma condition_p
def to_prevalence_dataframe(df, pop=None, dc=None, entities=None, methods=None, age_bands=None,
                            condition_prevalences=None, herds=None,
                            latest=False, overall=False, condition_p=(MA_CONDITION_P + NYC_CONDITION_P) / 2):
    df = df.loc[:, ['date', 'entity', 'deaths', 'cases', 'tests', 'population']]
    if entities:
        entities = [entities] if isinstance(entities, str) else entities
        df = df.loc[df['entity'].isin(entities), :]

    df = add_prevalence_dimension(df)
    if methods:
        methods = [methods] if isinstance(methods, str) else methods
        df = df.loc[df['method'].isin(methods), :]

    if age_bands:
        df = add_age_dimension(df, pop=pop, dc=dc, groupby=['date', 'entity'])
        age_bands = [age_bands] if isinstance(age_bands, str) else age_bands
        df = df.loc[df['age_band'].isin(age_bands), :]

    if condition_prevalences:
        df = add_condition_dimension(df, p=condition_p)
        df = df.loc[df['condition_prevalence'].isin(condition_prevalences), :]

    if herds:
        df = add_herd_dimension(df)
        df = df.loc[df['herd'].isin(herds), :]

    groupby_sub = (['method'] +
                   (['age_band'] if age_bands else []) +
                   (['herd'] if herds else []) +
                   (['condition', 'condition_prevalence'] if condition_prevalences else []))
    if latest:
        df = df.groupby(['entity'] + groupby_sub).last().reset_index()

    if overall:
        # print(f'groupby_sub: {groupby_sub}')
        df = df.groupby(['date'] + groupby_sub)[['deaths', 'cases', 'tests', 'population']].sum().reset_index()

    df = add_prevalence_cols(df)
    if herds:
        df = add_herd_cols(df)

    if round:
        for col in ['deaths', 'cases', 'tests', 'population',
                    'herd_cases', 'herd_deaths', 'herd_deaths_per_million']:
            if col in df.columns:
                df[col] = df[col].round()

    return df


def add_infections_dimension(df):
    '''
    Add estimates of number of infections based on antibody serology prevalence studies.
    :param df:
    :return:
    '''
    methods = ['confirmed', 'ny_confirmed', 'miami_confirmed', 'az_confirmed']
    entities = df['entity'].unique()
    ents, mets = list(zip(*itertools.product(entities, methods)))
    dm = pd.DataFrame({'entity': ents, 'method': mets})
    df = df.merge(dm, on='entity')

    sero = load_seroprevalence()

    idx = df['method'] == 'confirmed'
    df.loc[idx, 'infections'] = df.loc[idx, 'cases']

    idx = df['method'] == 'ny_confirmed'
    ny_factor = sero.loc[sero['entity'] == 'New York', 'factor'].values[0]
    df.loc[idx, 'infections'] = (ny_factor * df.loc[idx, 'cases']).round()

    miami_factor = sero.loc[sero['entity'] == 'Miami-Dade County', 'factor'].values[0]
    idx = df['method'] == 'miami_confirmed'
    df.loc[idx, 'infections'] = (miami_factor * df.loc[idx, 'cases']).round()

    az_factor = sero.loc[sero['entity'] == 'Arizona', 'factor'].values[0]
    idx = df['method'] == 'az_confirmed'
    df.loc[idx, 'infections'] = (az_factor * df.loc[idx, 'cases']).round()

    return df


def add_ifr_cols(df):
    """
    Add prevalence, infection fatality rate, case fatality rate and mortality rate columns.
    :param df: deaths cases infections population
    :return:
    """
    df['prevalence'] = df['infections'] / df['population']
    df['ifr'] = df['deaths'] / df['infections']
    df['cfr'] = df['deaths'] / df['cases']
    df['mr'] = df['deaths'] / df['population']
    return df


def add_age_dimension(df, age_methods=None, pops=None, dcs=None):
    '''
    Add an age_band dimension/axis/column. Within each entity and date,
    deaths, cases, tests, and population are distributed as follows:

    - population is replaced by the age_band population estimates from the US Census.
      For some age_bands that do not align with the 5-year bands of the census data,
      population estimates have been linearly interpolated. For example,
      the age_bad 0-17 in NY is comprised of the bands 0-4, 5-9, 10-14, and 3/5 of 15-19.
    - deaths are allocate according to known proportions of deaths by age band.
      The age_method column describes the state used to determine the distibution.
    - cases are allocated according to known proportions of cases by age band.
      The age_method column describes the state used to determine the distibution.
    - tests are allocated on a per capita basis, according to the population
      of the age band. This is inexact, but tests data is not used to calculate by age the
      prevalence or infection fatality rate or mortality rate of coronavirus.
    - infections are allocated on a per capita basis. This may be less accurate in
      states with major outbreaks in nursing homes. Seasonal influenza illness estimates are
      distributed evenly across age to some extent. Estimates are within a factor of 2 more or less.
      The 95% confidence intervals of estimates overlap or almost overlap.

    :param df: entity date deaths cases tests population infections.
    :param pops: list of dataframes. entity age_band population
    :param dcs: list of dataframes. entity age_band deaths cases. Only for a single entity.
    :return: add age_band dimension. distribute deaths, cases, tests, population across age bands.
    '''
    # load data if not provided.
    # this is slow, so passing pops and dcs is faster.
    if pops is None:
        pop_nyc = load_census_state_population_by_age_data(bands='nyc')
        pop_ma = load_census_state_population_by_age_data(bands='ma')
        pop_ri = pop_ma
        pops = [pop_nyc, pop_ma, pop_ri]

    if dcs is None:
        _, dc_nyc = load_nyc_age_comorbidity_death_data()
        dc_ma = load_ma_data_by_date_age().groupby('age_band').last().reset_index()
        dc_ri = load_ri_data_by_date_age(bands='ma').groupby('age_band').last().reset_index()
        dcs = [dc_nyc, dc_ma, dc_ri]

    age_methods = ['ny_age', 'ma_age', 'ri_age'] if age_methods is None else age_methods

    dgs = []
    for age_method, pop, dc in zip(age_methods, pops, dcs):
        dg = df.copy()
        dg['age_method'] = age_method
        # population
        dg = dg.merge(pop, on='entity', suffixes=('', '_age'))  # broadcast age_band population data across entities
        dc = dc.loc[:, ['age_band', 'deaths', 'cases']]
        dg = dg.merge(dc, on='age_band', suffixes=('', '_dc'))  # broadcast deaths_age, cases_age across entities, methods, dates.

        # population
        groupby = ['entity', 'date','method']
        dg['population_age_frac'] = dg['population_age'] / dg.groupby(groupby)['population_age'].transform(sum)
        dg['population_age2'] = dg['population_age_frac'] * dg['population']

        # tests
        # methods: distribute tests uniformly across population.
        # since we don't have tests segmented by age for any states currently.
        # DATA NEEDED: tests segmented by age for at least one state.
        dg['tests_age'] = dg['population_age_frac'] * dg['tests']

        # infections
        # methods: distribute infections uniformly across population.
        # pros: influenza illness are distributed uniformly (within a factor of 2)
        # cons: outbreaks in nursing homes, closures of schools, etc., might make distribution look
        #   different from influenza.
        dg['infections_age'] = dg['population_age_frac'] * dg['infections']

        # cases
        # methods: distribute cases according to the distribution of cases across age bands (in NYC data)
        # an alternative would be to distribute cases uniformly across population.
        dg['cases_age_frac'] = dg['cases_dc'] / dg.groupby(groupby)['cases_dc'].transform(sum)
        dg['cases_age'] = dg['cases_age_frac'] * dg['cases']

        # deaths
        # methods: distribute cases according to the distribution of cases across age bands (in NYC data)
        # an alternative would be to distribute cases uniformly across population.
        dg['deaths_age_frac'] = dg['deaths_dc'] / dg.groupby(groupby)['deaths_dc'].transform(sum)
        dg['deaths_age'] = dg['deaths_age_frac'] * dg['deaths']

        dgs.append(dg)

    dg = dgs[0]
    for d in dgs[1:]:
        dg = dg.append(d)

    # replace original columns with age-band values
    dg = dg.drop(columns=['population', 'tests', 'cases', 'deaths',
                          'infections', 'deaths_dc', 'cases_dc',
                          'population_age_frac', 'population_age', 'cases_age_frac', 'deaths_age_frac'])
    dg = dg.rename(columns={'population_age2': 'population',
                            'tests_age': 'tests',
                            'cases_age': 'cases',
                            'deaths_age': 'deaths',
                            'infections_age': 'infections'
                            })
    return dg


def add_prevalence_dimension(df, methods=['confirmed', 'adjusted_confirmed', 'erickson', 'adjusted_erickson', 'miami_confirmed']):
    '''
    Add an axis/dimension for prevalence methods
    :param df:
    :return:
    '''
    entities = df['entity'].unique()
    ents, mets = list(zip(*itertools.product(entities, methods)))
    dm = pd.DataFrame({'entity': ents, 'method': mets})
    dg = df.merge(dm, on='entity')
    return dg


def add_age_dimension_old(df, pop, dc, groupby=['entity', 'date', 'method']):
    '''
    Add an age_band dimension/axis and divide deaths, cases, tests, and population across the dimension.

    We divide those counts according to data and simple models.

    We have data for deaths and cases by state and age band
    We have data for population by state and age band
    We have data for deaths and cases by state and date.
    We would like to have: tests by state and age band (and by date would be nice!).
    We would love to have: comorbidities by state and age band (and by date if we can dream).

    :param df: entity [date] [method] deaths cases tests.
    :param pop: entity age_band population
    :param dc: entity age_band deaths cases. Only for a single entity. To be broadcast across entities.
    :param groupby: population_age_frac is determined by summing population_age across the age dimension.
      groupby is a list of other dimensions in df, like entity, date, method, herd, or condition_prevalence.
    :return: add age_band dimension. distribute deaths, cases, tests, population across age bands.
    '''
    dc = dc[dc['entity'] == dc['entity'].iloc[0]]  # insist on only a single entity in dc.
    dg = df.merge(pop, on='entity', suffixes=('', '_age'))  # broadcast population_age across dates, methods
    dg = dg.merge(dc, on='age_band', suffixes=('', '_dc'))  # broadcast deaths_age, cases_age across entities, methods, dates.

    # methods: distribute tests uniformly across population.
    # since we don't have tests segmented by age for any states currently.
    # DATA NEEDED: tests segmented by age for at least one state.
    dg['population_age_frac'] = dg['population_age'] / dg.groupby(groupby)['population_age'].transform(sum)
    dg['tests_age'] = dg['population_age_frac'] * dg['tests']  # we don't have tests by age and state.

    # methods: distribute cases according to the distribution of cases across age bands (in NYC data)
    # an alternative would be to distribute cases uniformly across population.
    dg['cases_age_frac'] = dg['cases_dc'] / dg.groupby(groupby)['cases_dc'].transform(sum)
    dg['cases_age'] = dg['cases_age_frac'] * dg['cases']  # methods: uses cases * the proportion of cases from the age band

    # methods: distribute cases according to the distribution of cases across age bands (in NYC data)
    # an alternative would be to distribute cases uniformly across population.
    dg['deaths_age_frac'] = dg['deaths_dc'] / dg.groupby(groupby)['deaths_dc'].transform(sum)
    dg['deaths_age'] = dg['deaths_age_frac'] * dg['deaths']  # methods: uses cases * the proportion of cases from the age band

    dg = dg.drop(columns=['population', 'tests', 'cases', 'deaths', 'deaths_dc', 'cases_dc'])
    dg = dg.rename(columns={'population_age': 'population',
                            'tests_age': 'tests',
                            'cases_age': 'cases',
                            'deaths_age': 'deaths'})
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


def add_condition_dimension(df, p=0.985):
    '''
    :param df:
    :param p: the probability that a person who died of coronavirus has a comorbidity.
    :return:
    '''
    prevalences = [0.1, 0.3, 0.5]
    conditions = [False, True]
    entities = df['entity'].unique()
    ents, pres, cons = list(zip(*itertools.product(entities, prevalences, conditions)))
    dm = pd.DataFrame({'entity': ents, 'condition_prevalence': pres, 'condition': cons})
    df = df.merge(dm, on=['entity'])

    idx = df['condition'] == True
    df.loc[idx, 'deaths'] = p * df.loc[idx, 'deaths']
    df.loc[idx, 'cases'] = df.loc[idx, 'condition_prevalence'] * df.loc[idx, 'cases']
    df.loc[idx, 'tests'] = df.loc[idx, 'condition_prevalence'] * df.loc[idx, 'tests']
    df.loc[idx, 'population'] = df.loc[idx, 'condition_prevalence'] * df.loc[idx, 'population']

    idx = df['condition'] == False
    df.loc[idx, 'deaths'] = (1 - p) * df.loc[idx, 'deaths']
    df.loc[idx, 'cases'] = (1 - df.loc[idx, 'condition_prevalence']) * df.loc[idx, 'cases']
    df.loc[idx, 'tests'] = (1 - df.loc[idx, 'condition_prevalence']) * df.loc[idx, 'tests']
    df.loc[idx, 'population'] = (1 - df.loc[idx, 'condition_prevalence']) * df.loc[idx, 'population']
    return df


def add_herd_cols(df):
    '''
    Given an infection fatality rate (ifr), a population, and a prevalence,
    compute the number of cases, deaths and deaths per million.
    :param df: herd, population, ifr
    :return:
    '''
    df['herd_cases'] = df['herd'] * df['population']
    df['herd_deaths'] = df['herd_cases'] * df['ifr']
    df['herd_deaths_per_million'] = 1000000 * df['herd_deaths'] / df['population']
    return df


def add_prevalence_cols(df):
    """
    Add prevalence (and optionally prevalence_age) for every prevalence method.
    :param df: method tests cases deaths population [tests_age cases_age deaths_age population_age]
    :return:
    """
    erickson_adjustment, confirmed_adjustment, miami_adjustment = seroprevalence_adjustments()
    idx = df['method'] == 'confirmed'
    df.loc[idx, 'prevalence'] = df.loc[idx, 'cases'] / df.loc[idx, 'population']
    idx = df['method'] == 'adjusted_confirmed'
    df.loc[idx, 'prevalence'] = confirmed_adjustment * df.loc[idx, 'cases'] / df.loc[idx, 'population']
    idx = df['method'] == 'miami_confirmed'
    df.loc[idx, 'prevalence'] = miami_adjustment * df.loc[idx, 'cases'] / df.loc[idx, 'population']
    idx = df['method'] == 'erickson'
    df.loc[idx, 'prevalence'] = df.loc[idx, 'cases'] / df.loc[idx, 'tests']
    idx = df['method'] == 'adjusted_erickson'
    df.loc[idx, 'prevalence'] = erickson_adjustment * df.loc[idx, 'cases'] / df.loc[idx, 'tests']

    df['ifr'] = df.loc[:, 'deaths'] / (df.loc[:, 'prevalence'] * df.loc[:, 'population'])
    df['cfr'] = df['deaths'] / df['cases']
    return df


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
    age_bands = ['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
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


def load_ri_data_by_date_age(bands=None):
    """
    Source: https://health.ri.gov/covid/
    Source: https://docs.google.com/spreadsheets/d/1n-zMS9Al94CPj_Tc3K7Adin-tN9x1RSjjx2UzJ4SV7Q/edit#gid=771623753

    :param bands: if 'ma' return stats aggregated by MA age bands
    :return: df with date entity age_band cases deaths hospitalized
    """
    filename = DATA_DIR / 'ri_data_by_date_age_20200521.csv'
    # the date the data was copied from the spreadsheet
    data_20200521 = [
        ['0-9', 277, 5, 0],
        ['10-19', 602, 9, 0],
        ['20-29', 1915, 51, 4], # <5 (not 4) is the actual number in the spreadsheet. WTF?
        ['30-39', 2029, 109, 6],
        ['40-49', 1996, 170, 6],
        ['50-59', 2118, 266, 28],
        ['60-69', 1509, 325, 62],
        ['70-79', 1012, 275, 142],
        ['80-89', 929, 202, 170],
        ['90-99', 602, 82, 132],
        ['100+', 44, 4, 9] # <5 (not 4) is the actual number in the spreadsheet.
    ]
    df = pd.DataFrame(data_20200521, columns=['age_band', 'cases', 'hospitalized', 'deaths'])
    df['entity'] = 'Rhode Island'
    df['date'] = pd.to_datetime('2020-05-21')

    if bands == 'ma':
        band0 = df.loc[df['age_band'].isin(['0-9', '10-19']), :].groupby(['entity', 'date'])[
            ['cases', 'hospitalized', 'deaths']].sum().reset_index()
        band0['age_band'] = '0-19'
        band20 = df.loc[df['age_band'] == '20-29']
        band30 = df.loc[df['age_band'] == '30-39']
        band40 = df.loc[df['age_band'] == '40-49']
        band50 = df.loc[df['age_band'] == '50-59']
        band60 = df.loc[df['age_band'] == '60-69']
        band70 = df.loc[df['age_band'] == '70-79']
        band80 = df.loc[df['age_band'].isin(['80-89', '90-99', '100+']), :].groupby(['entity', 'date'])[
            ['cases', 'hospitalized', 'deaths']].sum().reset_index()
        band80['age_band'] = '80+'
        df = (band0.append(band20).append(band30).append(band40).append(band50)
              .append(band60).append(band70).append(band80))

    return df.loc[:, ['entity', 'date', 'age_band', 'cases', 'deaths', 'hospitalized']]


def load_ma_data_by_date_age(cache=False):
    '''
    Return daily counts by age band for cases, hospitalized, deaths for the entity 'Massachusetts'.
    Downloaded on 05/07 (covering up to 05/06).

    To Do: Automatic download would be possible. download, unzip folder, read csv.

    :param cache:
    :return: a dataframe with entity, date, age_band dims and cases, deaths, hospitalized counts.
    '''
    filename = DATA_DIR / 'ma_data_by_date_age_20200506.csv'
    today = datetime.date.today()
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    url, yester_url = [f'https://www.mass.gov/doc/covid-19-raw-data-{d.strftime("%b").lower()}' +
                       f'-{d.day}-{d.year}/download' for d in [today, yesterday]]
    # print(f'load_ma_data_by_age url: {url}\nyester_url: {yester_url}')
    df = pd.read_csv(filename)
    df = df.rename(columns={
        'Date': 'date',
        'Cases': 'cases',
        'Deaths': 'deaths',
        'Age': 'age_band',
        'Hospitalized': 'hospitalized',
    })
    df['date'] = pd.to_datetime(df['date'])
    df['entity'] = 'Massachusetts'
    df = df.loc[df['age_band'] != 'Unknown', :].sort_values(['date', 'age_band'])
    return df


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
                              'deaths': deaths_20200501})
    df = pd.DataFrame({'age_band': age_bands,
                           'min_age': min_age,
                           'max_age': max_age,
                       'cases': cases_20200501,
                       'deaths': list(age_comorbidity_df.groupby('age_band').sum()['deaths'])})
    df = df.loc[df['age_band'] != 'unknown', :]
    # df['cumulative_deaths'] = df['deaths'].cumsum()
    # df['cumulative_death_pct'] = 100 * df['cumulative_deaths'] / df['deaths'].sum()
    # df['cumulative_cases'] = df['cases'].cumsum()
    # df['cumulative_cases_pct'] = 100 * df['cumulative_cases'] / df['cases'].sum()
    df['deaths_per_case'] = df['deaths'] / df['cases']
    # df['cumulative_deaths_per_case'] = df['cumulative_deaths'] / df['cumulative_cases']
    df['entity'] = 'New York'
    df['date'] = pd.to_datetime('2020-05-01')

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


def load_cdc_influenza_burden_by_age():
    # print('Table: Estimated rates of influenza disease outcomes, per 100,000 by age group')
    # print('United States, 2016-2017 influenza season')
    cols = ['Age group', 'Illness Rate Estimate', 'Illness Rate 95% Cr I',
            'Medical Visit Rate Estimate', 'Medical Visit Rate 95% Cr I',
            'Hospitalization Rate Estimate', 'Hospitalization Rate 95% Cr I',
            'Mortality Rate Estimate', 'Mortality Rate 95% Cr I']
    data = [['0-4 yrs',
             11949.70, (6976.6, 63240.0),
             8006.30, (4638.7, 42783.7),
             83.3, (48.6, 440.9),
             0.6, (0.0, 1.5)],
            ['5-17 yrs',
             12011.70, (7932.3, 24567.3),
             6246.10, (4080.4, 12787.9),
             32.9, (21.7, 67.4),
             0.2, (0.0, 0.8)],
            ['18-49 yrs',
             6786.10, (4858.3, 11261.3),
             2510.90, (1731.2, 4245.5),
             38.1, (27.3, 63.2),
             1, (0.6, 1.5)],
            ['50-64 yrs',
             11766.10, (8375.6, 20692.7),
             5059.40, (3514.3, 8875.9),
             124.8, (88.8, 219.4),
             6, (4.5, 7.8)],
            ['65+ yrs',
             7404.30, (4807.1, 14855.1),
             4146.40, (2635.0, 8439.0),
             673.1, (437.0, 1350.5),
             66.7, (47.2, 112.3)]]
    df = pd.DataFrame(data, columns=cols)
    df['season'] = '2016-2017'

    data = [
        ['0-4 yrs',
         18448.1, (12856.5, 36475.0),
         12360.2, (8501.3, 24596.7),
         128.6, (89.6, 254.3),
         0.6, (0.0, 1.8)],
        ['5-17 yrs',
         13985.6, (10983.6, 18987.0),
         7272.5, (5589.3, 9972.2),
         38.3, (30.1, 52.1),
         1.0, (0.4, 2.6)],
        ['18-49 yrs',
         10469.7, (8895.6, 14075.1),
         3873.8, (3092.9, 5321.7),
         58.8, (49.9, 79.0),
         2.0, (1.2, 5.0)],
        ['50-64 yrs',
         20881.1, (14828.2, 36378.8),
         8978.9, (6145.3, 15818.0),
         221.4, (157.2, 385.8),
         10.6, (6.7, 25.0)],
        ['65+ yrs',
         11690.6, (7682.1, 23175.5),
         6546.7, (4207.2, 13023.8),
         1062.8, (698.4, 2106.9),
         100.1, (70.8, 163.7)]]
    dg = pd.DataFrame(data, columns=cols)
    dg['season'] = '2017-2018'

    return df.append(dg).reset_index(drop=True)


def main():
    print(map_owid_to_un_pop())


if __name__ == '__main__':
    main()
