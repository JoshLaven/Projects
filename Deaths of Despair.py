import requests
import pandas as pd
import datetime
import numpy as np
import pandas_datareader.data as web
from bs4 import BeautifulSoup as bs
import os
import matplotlib.pyplot as plt
from functools import reduce
import statsmodels.formula.api as sm
from statsmodels.iolib.summary2 import summary_col

#Base Path
base_path = os.getcwd()

#Creating generalized data search.
#Search words can be added as values in the dictionary and any
#datasets with those keywords in the repository will be downloaded.
repo_url = 'https://github.com/owid/owid-datasets/tree/master/datasets'
raw_repo_url = 'https://raw.githubusercontent.com'
report_params = {'RepoSearchWord1':'Opioid',
                 'RepoSearchWord2':'Suicide',
                 'RepoSearchWord3': 'suicide',
                 'FredSearchWord': 'UNRATE'}



def get_Gitlinks(word1, word2, word3):
#Get the links of desired datasets in repository
    response = requests.get(repo_url)
    soup = bs(response.text,'lxml')
    dataset_dict = {}

    for link in soup.find_all('a',{'class':'js-navigation-open'}):
        if (word1 in link['href']) or (word2 in link['href']) or (word3 in link['href']):
            href = link.get('href')
            title = link.get('title') + ".csv"
            dataset_dict[href]= title

    return dataset_dict

def edit_href(hreflink):
#Edit links from Soup into the appropriate format for the get request
    pieces = hreflink.split('/')
    new_link = ''

    for a in pieces:
        if a !='tree':
            new_link = new_link + '/' + a

    new_link = raw_repo_url+new_link[1:] + '/' + pieces[6] +'.csv'

    return new_link


def get_Git_data():
    link_dict = get_Gitlinks(report_params['RepoSearchWord1'], report_params['RepoSearchWord2'], report_params['RepoSearchWord3'])

    final_linknames= {edit_href(k):v for k,v in link_dict.items()}


    for i, j in final_linknames.items():
        response = requests.get(i)
        data = response.text
        with open(os.path.join(base_path,j), 'w') as ofile:
            ofile.write(data)

def get_FRED_data():
    start = datetime.date(year=1950, month=1,  day=1)
    end = datetime.date(year=2015, month=12, day=31)
    series = report_params['FredSearchWord']
    source = 'fred'
    data = web.DataReader(series, source, start, end)
    data.to_csv(os.path.join(base_path,'FRED Data.csv'))

get_Git_data()
get_FRED_data()

def check_data():
#Checks CSVs that were written to the directory for naming purposes
    for f in os.listdir(base_path):
        if f.endswith('.csv'):
            print (f)
check_data()

#Read in Data
df_allsuiciderates = pd.read_csv(os.path.join(base_path,'Age-adjusted suicide rates, 1950-2005 - WHO (2005).csv'))
df_ussuiciderates2 = pd.read_csv(os.path.join(base_path,'Suicide rates in the United States - AFSP (2017).csv'))
df_opioid = pd.read_csv(os.path.join(base_path,'Opioid deaths due to overuse in the US - CDC WONDER (2017).csv'))
df_unempraw = pd.read_csv(os.path.join(base_path,'FRED Data.csv'),parse_dates=['DATE'],infer_datetime_format=True)


#Getting Datasets ready for merge
def percent_change(df,column):
#Function to add columns for percentage changes and log values
    df[column + ' % Change'] = df[column].pct_change(periods = 1)
    df[column + ' % Change']= df[column + ' % Change'].fillna(0)
    df[column + ' Log'] = np.log(df[column])
    return df

def us_mergesuicide(df_allsuiciderates):
#Merging the US dataframe that is 1950-2005 with the one that is 2006-2015
    df_ussuiciderates1 = df_allsuiciderates.loc[(df_allsuiciderates["Entity"] == 'United States')]
    df_ussuiciderates1 = df_ussuiciderates1.iloc[:,0:3]
    #Rename Columns
    df_ussuiciderates1.rename(columns={'Suicide rate (WHO (2005))':'Suicides'}, inplace=True)
    df_ussuiciderates2.rename(columns={'Suicide Rate (AFSP (2017))':'Suicides'}, inplace=True)

    #Merging all US suicide data
    df_ussuiciderates_total = pd.concat([df_ussuiciderates1,df_ussuiciderates2], ignore_index=True)
    return df_ussuiciderates_total

def unemp_format(frame):
#Formatting Unemployment dataframe for easier merge
    frame['Year']= frame['DATE'].apply(lambda x: x.year)
    frame = frame.groupby('Year', as_index=False)['UNRATE'].mean()
    frame['Entity']='United States'
    frame.rename(index = str, columns={'UNRATE':'Unemployment Rate'}, inplace = True)
    return frame

def merge_unemp_suicide(df1, df2):
#Merging unemployment rate and suicide rate to save in its own workbook
#because both have data going back to 1950
    df_unemp_suicide = pd.merge(df1, df2, on=['Year', 'Entity'], how='left')
    percent_change(df_unemp_suicide,'Suicides')
    percent_change(df_unemp_suicide,'Unemployment Rate')
    return df_unemp_suicide

#Dataframes ready for merging
df_ussuiciderates_total = us_mergesuicide(df_allsuiciderates)
df_unemp = unemp_format(df_unempraw)
df_opioid.rename(columns={'Total Overdose Deaths (CDC WONDER (2017))':'Opioid Overdose Deaths'}, inplace = True)
df_unemp_suicide = merge_unemp_suicide(df_ussuiciderates_total, df_unemp)


def all_merge():
#Merge and finalize all 3 Dfs
    alldata = reduce(lambda x,y: pd.merge(x,y, on=['Year', 'Entity'], how='inner'), [df_opioid, df_ussuiciderates_total,df_unemp])
    alldata.set_index('Year', inplace = True)
    percent_change(alldata,'Suicides')
    percent_change(alldata,'Unemployment Rate')
    percent_change(alldata,'Opioid Overdose Deaths')
    return alldata

#Final Data
df_alldata = all_merge()


def save_data():
#Writes final data to csv
    df_alldata.to_csv(os.path.join(base_path,'Final Data - All.csv'))
    df_unemp_suicide.to_csv(os.path.join(base_path,'Unemployment and Suicide Data.csv'))

save_data()

def indiv_plots():
    #Graph of Suicide Rate v Year
    df_ussuiciderates_total.plot(title = "Number of Suicides per 100,000 People",x='Year', y='Suicides')
    plt.savefig(os.path.join(base_path, 'Suicides in the US'))
    #Graph of Unemployment Rate per Year
    df_unemp.plot(title = "Average Yearly Unemployment Rate in the US", x='Year', y= 'Unemployment Rate')
    plt.savefig(os.path.join(base_path, 'Unemployment Rate in the US'))
    #Graph of Opioid Overdoses per Year)
    df_alldata.plot(title = "Number of Opioid Overdose Deaths per 100,000 People", y='Opioid Overdose Deaths')
    plt.savefig(os.path.join(base_path, 'Opioid Overdose Deaths in the US'))

indiv_plots()

#Graph of unemployment rate and suicide rate per year
def unemp_suicide_graph():
    #Merge Unemployment and Suicide datasets going back to 1950
    df_unemp_suicide.set_index('Year', inplace = True)
    #Create plot
    fig, ax = plt.subplots(figsize = (10,6))
    plt.title("Unemployment Rate and Suicides")
    plt.xlabel('Year')
    plt.plot(df_unemp_suicide['Unemployment Rate'], '-b')
    plt.plot(df_unemp_suicide['Suicides'], '-r')
    plt.legend(['Unemployment Rate (%)','Suicides per 100,000 People'])
    plt.savefig(os.path.join(base_path, 'Unemployment Rate and Suicides since 1950'))
    plt.show()

unemp_suicide_graph()

#Graph of all three
def alldata_graph():
    fig, ax = plt.subplots(figsize = (10,6))
    plt.title("US Statistics")
    ax.spines['left'].set_color('red')
    plt.xlabel('Year')
    plt.ylabel('Per 100,000 People', color='r')
    plt.plot(df_alldata['Opioid Overdose Deaths'], '-r')
    plt.plot(df_alldata['Suicides'], '--r')
    plt.legend(['Opioid Overdose Deaths','Suicides'])
    #Make second axis
    ax2 = ax.twinx()
    plt.ylabel('Unemployment Rate (%)', color='b')
    plt.plot(df_alldata['Unemployment Rate'], '-b')
    ax2.spines['right'].set_color('blue')
    plt.legend()
    #save and close
    plt.savefig(os.path.join(base_path, 'All Data Plot'))
    plt.show()
    plt.close()

alldata_graph()

#Graph of all three (percentages)
def alldata_graph_pchange():
    fig, ax = plt.subplots(figsize = (10,5))
    plt.title("US Statistics - % Change")
    plt.xlabel('Year')
    plt.ylabel('% Change', color='k')
    plt.plot(df_alldata['Unemployment Rate % Change'], '-b')
    plt.plot(df_alldata['Opioid Overdose Deaths % Change'], '-r')
    plt.plot(df_alldata['Suicides % Change'], '--r', )
    ax.legend(['Unemployment Rate','Opioid Overdose Deaths','Suicides'],loc=2)
    #Save and Close
    plt.savefig(os.path.join(base_path, 'All Data Plot - Percent Change'))
    plt.show()
    plt.close()

alldata_graph_pchange()

def reg_eq(df,yvar,xvar):
    df['Intercept']=1
    model = sm.OLS(df[yvar],df[['Intercept',xvar]])
    return model

def run_regression():
    #Suicide and Unemployment Rate Regression
    Result1 = reg_eq(df_unemp_suicide, 'Suicides', 'Unemployment Rate').fit()
    Result2 = reg_eq(df_unemp_suicide,'Suicides Log','Unemployment Rate').fit()
    #Suicide and Unemployment Rate Regression - Past 16 Years
    Result3 = reg_eq(df_alldata,'Suicides Log','Unemployment Rate').fit()

    #Opioid Overdose Deaths and Unemployment Rate Regression
    Result4 = reg_eq(df_alldata,'Opioid Overdose Deaths','Unemployment Rate').fit()
    Result5 = reg_eq(df_alldata,'Opioid Overdose Deaths Log','Unemployment Rate').fit()

    #Suicides and Opioid Overdose Deaths
    Result6 = reg_eq(df_alldata,'Suicides','Opioid Overdose Deaths').fit()

    RegOutput = summary_col([Result1,Result2,Result3,Result4,Result5, Result6], stars=True)

    return RegOutput

def Regression_Output():
    RegResults = run_regression().as_text()
    resultFile = open(os.path.join(base_path,'Regression Results.txt'),'w')
    resultFile.write(RegResults)
    resultFile.close()

Regression_Output()
