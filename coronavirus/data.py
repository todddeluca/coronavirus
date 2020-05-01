import datetime
import numpy as np
import pandas as pd
from pathlib import Path


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


def before_threshold(df, col, thresh):
    '''
    Return boolean index of df rows before the first day when the threshold
    was met, i.e. when df[col] >= thresh. This is done with each entity.
    :param df: entity/state/country column, ordered by date column, cases, deaths, etc.
    '''
    return df.groupby(['entity'])[col].transform(lambda s: (s >= thresh).cumsum() == 0)


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


'''
Data on Age, Comorbidity and Death
New York City Sources:
- NYC Health Department: https://www1.nyc.gov/site/doh/covid/covid-19-data.page
- deaths by age and comorbidity status: https://www1.nyc.gov/assets/doh/downloads/pdf/imm/covid-19-daily-data-summary-deaths-04222020-1.pdf
- cases by age band: https://www1.nyc.gov/assets/doh/downloads/pdf/imm/covid-19-daily-data-summary-04222020-1.pdf
'''

def load_nyc_age_comorbidity_death_data(cache=False):
    '''
    Return a tidy table containing deaths segmented by age band and comorbidity status.
    '''
    age_bands = ['0-17', '18-44', '45-64', '65-74', '75 and over', 'unknown']
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
    cases_20200421 = [2839, 51217, 50694, 17474, 15932, 279]
    cases_20200426 = [3488, 57462, 57556, 19545, 17735, 314]
    deaths_df = pd.DataFrame({'age_band': np.repeat(age_bands, len(comorbidity)),
                              'comorbidity': comorbidity * len(age_bands),
                              'deaths': deaths_20200426})
    df = pd.DataFrame({'age_band': age_bands,
                       'cases': cases_20200426,
                       'deaths': list(deaths_df.groupby('age_band').sum()['deaths'])})
    if cache:
        pd.to_csv(DATA_DIR / 'nyc_age_comorbidity_deaths.csv')

    return deaths_df, df


def main():
    print(map_owid_to_un_pop())


if __name__ == '__main__':
    main()
