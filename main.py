import time

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
from flask import Flask, jsonify, url_for, request
from flask import render_template
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import datetime
import re
import json

app = Flask(__name__, static_url_path='/static')
df = pd.read_csv('data/US_Accidents_Dec20_copy.csv')
df = df.sample(300000)
df['End_Time'] = pd.to_datetime(df['End_Time'])

current_df = df

current_state = None
current_start_time = None
current_end_time = None

us_state_abbrev_reverse = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands': 'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

us_state_abbrev = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming',
}


def get_state_time(time_range, state=''):
    month = {'1': 'January', '2': 'February', '3': 'March', '4': 'April', '5': 'May', '6': 'June', '7': 'July',
             '8': 'August', '9': 'September', '10': 'October', '11': 'November', '12': 'December'}
    day = {'0': 'Monday', '1': 'Tuesday', '2': 'Wednesday', '3': 'Thursday', '4': 'Friday', '5': 'Saturday',
           '6': 'Sunday'}
    year_dic = {'2016': 0, '2017': 0, '2018': 0, '2019': 0, '2020': 0}
    month_dic = {'January': 0, 'February': 0, 'March': 0, 'April': 0, 'May': 0, 'June': 0, 'July': 0, 'August': 0,
                 'September': 0, 'October': 0, 'November': 0, 'December': 0}
    day_dic = {'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0, 'Saturday': 0, 'Sunday': 0}
    hour_dic = {'00-02': 0, '02-04': 0, '04-06': 0, '06-08': 0, '08-10': 0, '10-12': 0, '12-14': 0, '14-16': 0, '16-18': 0, '18-20': 0, '20-22': 0, '22-00': 0}
    print(current_df)

    if (time_range == 'year'):
        tmp_data = current_df.groupby(current_df['End_Time'].dt.strftime('%Y')).count()
        tmp_year_dic = tmp_data.to_dict()
        for key, val in tmp_year_dic['ID'].items():
            year_dic[key] = val
        year_list = []
        for key, val in year_dic.items():
            tmp_dic = {}
            tmp_dic['key'] = key
            tmp_dic['value'] = val
            year_list.append(tmp_dic)
        return year_list
    elif (time_range == 'month'):
        tmp_data = current_df.groupby(current_df['End_Time'].dt.strftime('%B')).count()
        tmp_month_dic = tmp_data.to_dict()
        for key, val in tmp_month_dic['ID'].items():
            month_dic[key] = val
        month_list = []
        for key, val in month_dic.items():
            tmp_dic = {}
            tmp_dic['key'] = key
            tmp_dic['value'] = val
            month_list.append(tmp_dic)
        return month_list
    elif (time_range == 'day'):
        tmp_data = current_df.groupby(current_df['End_Time'].dt.strftime('%A')).count()
        tmp_day_dic = tmp_data.to_dict()
        for key, val in tmp_day_dic['ID'].items():
            day_dic[key] = val
        day_list = []
        for key, val in day_dic.items():
            tmp_dic = {}
            tmp_dic['key'] = key
            tmp_dic['value'] = val
            day_list.append(tmp_dic)
        return day_list
    elif (time_range == 'hour'):
        tmp_data = current_df.groupby(current_df['End_Time'].dt.strftime('%H')).count()
        tmp_hour_dic = tmp_data.to_dict()
        for key, val in tmp_hour_dic['ID'].items():
            if (int(key) < 2):
                hour_dic['00-02'] += val
            elif (int(key) > 2 and int(key) < 4):
                hour_dic['02-04'] += val
            elif (int(key) > 4 and int(key) < 6):
                hour_dic['04-06'] += val
            elif (int(key) > 6 and int(key) < 8):
                hour_dic['06-08'] += val
            elif (int(key) > 8 and int(key) < 10):
                hour_dic['08-10'] += val
            elif (int(key) > 10 and int(key) < 12):
                hour_dic['10-12'] += val
            elif (int(key) > 12 and int(key) < 14):
                hour_dic['12-14'] += val
            elif (int(key) > 14 and int(key) < 16):
                hour_dic['14-16'] += val
            elif (int(key) > 16 and int(key) < 18):
                hour_dic['16-18'] += val
            elif (int(key) > 18 and int(key) < 20):
                hour_dic['18-20'] += val
            elif (int(key) > 20 and int(key) < 22):
                hour_dic['20-22'] += val
            elif (int(key) > 22):
                hour_dic['22-00'] += val
        hour_list = []
        for key, val in hour_dic.items():
            tmp_dic = {}
            tmp_dic['key'] = key
            tmp_dic['value'] = val
            hour_list.append(tmp_dic)
        return hour_list


def get_time_for_particular_state(time_range, state):
    return get_state_time(time_range, state)


def get_time_for_all_states(time_range):
    ans = get_state_time(time_range)
    return ans

def get_county_frequency():
    tmp_df = current_df["County"].value_counts().rename_axis('county').reset_index(name='value')
    return tmp_df.to_json(orient="records")

def get_county_bar(state):
    data = current_df.loc[:, ['State', 'County']].values
    county_dic = {}
    count_list = []
    for val in data:
        if val[0] == state:
            count_list.append((val[1]))
            county_dic[val[1]] = 0
    for item in count_list:
        county_dic[item] += 1
    sorted_bar_dic = dict(sorted(county_dic.items(), key=lambda item: item[1], reverse=True))
    bar_list = []
    i = 0
    for key, val in sorted_bar_dic.items():
        if i == 12:
            break
        tmp_dic = {}
        tmp_dic["key"] = key
        tmp_dic["value"] = val
        bar_list.append(tmp_dic)
        i += 1
    # print(bar_list)
    return bar_list

def get_state_frequency():
    tmp_df = current_df["State"].value_counts().rename_axis('state').reset_index(name='value')
    tmp_df['state'] = tmp_df['state'].map(us_state_abbrev)
    return tmp_df.to_json(orient="records")

def get_bar_data():
    states = current_df['State'].tolist()
    list_set = set(states)
    unique_list = (list(list_set))
    # print(len(unique_list))
    bar_dic = {}
    for val in unique_list:
        bar_dic[val] = 0
    for val in states:
        bar_dic[val] += 1
    sorted_bar_dic = dict(sorted(bar_dic.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_bar_dic)
    i = 0
    bar_list = []
    for key, val in sorted_bar_dic.items():
        if i == 12:
            break
        tmp_dic = {}
        tmp_dic["key"] = key
        tmp_dic["value"] = val
        bar_list.append(tmp_dic)
        i += 1
    return bar_list

def get_time_series_data():
    tmp_df = current_df.loc[:, ["End_Time", "ID"]]
    tmp_df['End_Time'] = pd.to_datetime(tmp_df['End_Time'], errors='coerce')
    # column_names = df.columns.values
    # column_names[1] = 'Changed'
    # df.columns = column_names
    # return(tmp_df.groupby([tmp_df['Start_Time'].dt.year, tmp_df['Start_Time'].dt.month]).agg({'count'}))
    return tmp_df.resample('M', on="End_Time").agg({'count'})


def update_current_df(state, start_time=None, end_time=None):
    global current_df, current_state

    #Init
    if 'null' in state:
        current_df = df
        current_state = None
        return

    #State call
    if 'time' not in state:
        current_state = state
        current_df = df[df['State'].str.contains(us_state_abbrev_reverse[current_state])]
        return

    if current_state:
        current_df = df[df['State'].str.contains(us_state_abbrev_reverse[current_state])]
    else:
        current_df = df

    #Time call
    start_time = datetime.datetime.strptime(start_time[:10], "%Y-%m-%d")
    end_time = datetime.datetime.strptime(end_time[:10], "%Y-%m-%d")
    current_df['End_Time'] = pd.to_datetime(current_df['End_Time'])
    mask = (current_df['End_Time'] <= start_time) & (current_df['End_Time'] >= end_time)
    current_df = current_df.loc[mask]


def get_pcp_data():
    attributes = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Weather_Condition', 'Wind_Speed(mph)']
    df_tmp = current_df.sample(3000)
    df_pcp = df_tmp[attributes]
    data = df_pcp.values
    pcp_list = []
    i = 0
    for val in data:
        tmp_dic = {}
        tmp_dic[attributes[0]] = str(val[0])
        tmp_dic[attributes[1]] = str(val[1])
        tmp_dic[attributes[2]] = str(val[2])
        tmp_dic[attributes[3]] = str(val[3])
        tmp_dic[attributes[4]] = str(val[4])
        tmp_dic[attributes[5]] = str(val[5])
        # tmp_dic['label']=labels[i]
        # i+=1
        pcp_list.append(tmp_dic)
    return pcp_list


@app.route("/bar")
def bar():
    myList = get_bar_data()
    return jsonify(myList)


@app.route("/county/<state>")
def county(state):
    myList = get_county_bar(state)
    return jsonify(myList)


@app.route("/bar_state_year")
def bar_state_year():
    myList = get_time_for_all_states("year")
    return jsonify(myList)


@app.route("/bar_state_month")
def bar_state_month():
    myList = get_time_for_all_states("month")
    return jsonify(myList)


@app.route("/bar_state_day")
def bar_state_day():
    myList = get_time_for_all_states("day")
    return jsonify(myList)


@app.route("/bar_state_hour")
def bar_state_hour():
    myList = get_time_for_all_states("hour")
    return jsonify(myList)


@app.route("/time_series")
def time_series():
    df_ts = get_time_series_data()
    message = {}
    message["freq"] = df_ts["End_Time"]["count"].tolist()
    message["times"] = df_ts.index.strftime("%Y-%m").tolist()

    return message


@app.route("/chloropleth")
def state_frequency():
    myList = get_state_frequency()
    return myList


@app.route("/chloropleth-counties")
def county_frequency():
    myList = get_county_frequency()
    return myList


@app.route("/pcp")
def pcp_data():
    pcp = get_pcp_data()
    # print(pcp)
    return jsonify(pcp)


@app.route("/")
def d3_main():
    return render_template('index1.html')


@app.route("/update/<state>", methods=['POST'])
def d3_update(state):
    if request.get_json():
        update_current_df(state, request.get_json()['start_time'], request.get_json()['end_time'])
    else:
        update_current_df(state)
    return "HELLO-HERE"


if __name__ == "__main__":
    app.run(debug=True)
