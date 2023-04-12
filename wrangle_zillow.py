import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import env

import warnings
warnings.filterwarnings('ignore')

# query will grab our zillow data with notable specs from the exercise set, particularly:
# include logerror (we already joined on this table previously, no biggie)
# include date of transaction
# only include properties with lat+long filled
# narrow down to single unit properties (we will do this with pandas so we can investigate the fields)
query = '''
SELECT
-- everything from properties_2017 as prop:
    prop.*,
-- logerror and transactiondate from predictions_2017:
    predictions_2017.logerror,
    predictions_2017.transactiondate,
-- all the extra struff for left joins:
    air.airconditioningdesc,
    arch.architecturalstyledesc,
    build.buildingclassdesc,
    heat.heatingorsystemdesc,
    landuse.propertylandusedesc,
    story.storydesc,
    construct.typeconstructiondesc
-- FROM our first table, prperties 2017, as prop
FROM properties_2017 prop
-- We want to make sure that we are only grabbing units with a sale in 2017, and
-- most notably the transactions that happened most recently.
-- for every property, the maximum (most recent date and the property/parcelid
-- associated with that transaction.  This will give us a two-column table to
-- narrow down our prop table using an inner join.
JOIN (
    SELECT parcelid, MAX(transactiondate) AS max_transactiondate
    FROM predictions_2017
    GROUP BY parcelid
) pred USING(parcelid)
-- now that we have the properties table narrowed down to our specific parameters,
-- use those specific parameters to join both on parcelid and the max date.
-- we want to join on the instances that we narrowed down in the subquery. 
-- think about this as a join on a concatenated set of two columns.
JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                      AND pred.max_transactiondate = predictions_2017.transactiondate
                      -- everything else from here will be a left join, as we have the data set at the size we want and we dont want to make it smaller by reducing it based on missing columns in the following:
LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
LEFT JOIN storytype story USING (storytypeid)
LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
-- give me only properties where i have lattitude and longitude values:
WHERE prop.latitude IS NOT NULL
  AND prop.longitude IS NOT NULL
-- give me only properties with that max transaction date in 2017
  AND transactiondate <= '2017-12-31'
'''

def overview(df):
    '''
    print shape of DataFrame, .info() method call, and basic descriptive statistics via .describe()
    parameters: single pandas dataframe, df
    return: none
    '''
    print('--- Shape: {}'.format(df.shape))
    print('--- Info')
    df.info()
    print('--- Column Descriptions')
    print(df.describe(include='all'))

def nulls_by_columns(df):
    '''
    Get the number and proportion of values per column in the dataframe df

    parameters: single pandas dataframe, df
    return: none
    '''
    return pd.concat([
        df.isna().sum().rename('count'),
        df.isna().mean().rename('percent')
    ], axis=1)

def nulls_by_rows(df):
    '''
    Get the number and proportion of values per row in the dataframe df

    parameters: single pandas dataframe, df
    return: none
    '''
    return pd.concat([
        df.isna().sum(axis=1).rename('n_missing'),
        df.isna().mean(axis=1).rename('percent_missing'),
    ], axis=1).value_counts().sort_index()

def handle_missing_values(df, prop_required_column, prop_required_row):
    '''
    Utilizing an input proportion for the column and rows of DataFrame df,
    drop the missing values per the axis contingent on the amount of data present.
    '''
    n_required_column = round(df.shape[0] * prop_required_column)
    df = df.dropna(axis=1, thresh=n_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    return df

def acquire():
    '''
    aquire the zillow data utilizing the query defined earlier in this wrangle file.
    will read in cached data from any present "zillow.csv" present in the current directory.
    first-read data will be saved as "zillow.csv" following query.

    parameters: none

    '''
    if os.path.exists('zillow.csv'):
        df = pd.read_csv('zillow.csv')
    else:
        database = 'zillow'
        url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{database}'
        df = pd.read_sql(query, url)
        df.to_csv('zillow.csv', index=False)
    return df

def wrangle_zillow():
    '''
    acquires, gives summary statistics, and handles missing values contingent on
    the desires of the zillow data we wish to obtain.

    parameters: none
    return: single pandas dataframe, df
    '''
    # grab the data:
    df = acquire()
    # summarize and peek at the data:
    overview(df)
    nulls_by_columns(df).sort_values(by='percent')
    nulls_by_rows(df)
    # task for you to decide: ;)
    # determine what you want to categorize as a single unit property.
    # maybe use df.propertylandusedesc.unique() to get a list, narrow it down with domain knowledge,
    # then pull something like this:
    # df.propertylandusedesc = df.propertylandusedesc.apply(lambda x: x if x in my_list_of_single_unit_types else np.nan)
    # In our second iteration, we will tune the proportion and e:
    df = handle_missing_values(df, prop_required_column=.5, prop_required_row=.5)
    # take care of any duplicates:
    df = df.drop_duplicates()
    return df
    
