
## Introduction

Some coronavirus data visualization and analysis.


## Datasets

### Census data for state populations

- https://www.census.gov/programs-surveys/popest/data/data-sets.html
- http://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv?#

The file `data/nst-est2019-alldata.csv` contains 2019 state population estimates.


### Census data for US county populations

County-level 2019 Population estimate by age band.

- web page: https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html
- csv file: https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv

Files:
- `data/co-est2019-alldata.csv` contains county-level population data by age band?

Files:

- `data/cc-est2018-alldata.csv` contains county-level population data by age band.
- `data/cc-est2018-alldata.pdf` contains column descriptions


### 2010 Census data for urban population percentage by state

- https://www2.census.gov/geo/docs/reference/ua/PctUrbanRural_State.xls

The file `data/PctUrbanRural_State.csv` contains this 2010 percentage estimates.


### Coronavirus deaths data by state

https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv

Deaths and cases by state from the New York Times.

The file `data/us-states.csv` contains a local copy.

Deaths and cases by state and county from USAFacts.org. Source: https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/.

The files `data/covid_confirmed_usafacts.csv` and `data/covid_deaths_usafacts.csv` contain cases and deaths data respectively broken down by state and county.

The file `data/us_states_abbreviations.csv` maps state and territory abbreviations to state names. Source: https://github.com/aruljohn/us-states.


### US coronavirus deaths by age

CDC is publishing statistics for deaths broken down by age, location of death (e.g. hospital, longterm care facility, ...), sex and other variables. The url is https://data.cdc.gov/api/views/hc4f-j6nb/rows.csv?accessType=DOWNLOAD and the landing page is https://www.cdc.gov/nchs/nvss/vsrr/covid19/index.htm.


This table is saved in `cdc_provisional_coronavirus_deaths.csv`.

https://data.cdc.gov/api/views/hc4f-j6nb/rows.csv?accessType=DOWNLOAD&bom=true&format=true


### NYC coronavirus case and death statistics by age (and comorbidity)

NYC publishes stats on deaths and cases by age band.
They also publish stats on deaths by comorbity and age band status (but not cases.)

Data on Age, Comorbidity and Death
New York City Sources:
- NYC Health Department: https://www1.nyc.gov/site/doh/covid/covid-19-data.page
- deaths by age and comorbidity status: https://www1.nyc.gov/assets/doh/downloads/pdf/imm/covid-19-daily-data-summary-deaths-04222020-1.pdf
- cases by age band: https://www1.nyc.gov/assets/doh/downloads/pdf/imm/covid-19-daily-data-summary-04222020-1.pdf


### Coronavirus deaths by country

https://covid.ourworldindata.org/data/ecdc/full_data.csv

Daily deaths and cases. Total deaths and cases. For countries. From Our World In Data.

The file `data/full_data.csv` contains a local copy.


### State and Federal Actions

https://www.ncsl.org/research/health/state-action-on-coronavirus-covid-19.aspx


### Executive Orders

https://web.csg.org/covid19/executive-orders/


