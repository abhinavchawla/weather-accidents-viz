import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
from flask import Flask, jsonify, url_for
from flask import render_template
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import datetime
import re

app = Flask(__name__, static_url_path='/static')
df = pd.read_csv('data/US_Accidents_Dec20_copy.csv')
df = df.sample(3000)



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
    print(state)
    data = df.loc[:, ['State', 'Start_Time']].values
    month = {'1': 'January', '2': 'February', '3': 'March', '4': 'April', '5': 'May', '6': 'June', '7': 'July',
             '8': 'August', '9': 'September', '10': 'October', '11': 'November', '12': 'December'}
    day = {'0': 'Monday', '1': 'Tuesday', '2': 'Wednesday', '3': 'Thursday', '4': 'Friday', '5': 'Saturday',
           '6': 'Sunday'}
    year_dic = {'2016': 0, '2017': 0, '2018': 0, '2019': 0, '2020': 0}
    month_dic = {'January': 0, 'February': 0, 'March': 0, 'April': 0, 'May': 0, 'June': 0, 'July': 0, 'August': 0,
                 'September': 0, 'October': 0, 'November': 0, 'December': 0}
    day_dic = {'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0, 'Saturday': 0, 'Sunday': 0}
    hour_dic = {'00-06': 0, '06-12': 0, '12-20': 0, '20-00': 0}
    for val in data:
        if (state != ''):
            if (val[0] != state):
                continue
        date_time_str = val[1]
        match = re.match('\d\d\d\d-\d\d-\d\d\s+\d\d:\d\d:\d\d', date_time_str)
        date_time_str = match.group()
        date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
        year_dic[str(date_time_obj.year)] += 1
        month_dic[month[str(date_time_obj.month)]] += 1
        day_dic[day[str(date_time_obj.weekday())]] += 1
        if (0 < date_time_obj.hour < 6):
            hour_dic['00-06'] += 1
        elif (6 < date_time_obj.hour < 12):
            hour_dic['06-12'] += 1
        elif (12 < date_time_obj.hour < 20):
            hour_dic['12-20'] += 1
        elif (20 < date_time_obj.hour):
            hour_dic['20-00'] += 1
    if (time_range == 'year'):
        year_list = []
        for key, val in year_dic.items():
            tmp_dic = {}
            tmp_dic['key'] = key
            tmp_dic['value'] = val
            year_list.append(tmp_dic)
        return year_list
    elif (time_range == 'month'):
        month_list = []
        for key, val in month_dic.items():
            tmp_dic = {}
            tmp_dic['key'] = key
            tmp_dic['value'] = val
            month_list.append(tmp_dic)
        return month_list
    elif (time_range == 'day'):
        day_list = []
        for key, val in day_dic.items():
            tmp_dic = {}
            tmp_dic['key'] = key
            tmp_dic['value'] = val
            day_list.append(tmp_dic)
        return day_list
    elif (time_range == 'hour'):
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


def get_county_bar(state):
    data = df.loc[:, ['State', 'County']].values
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


def get_bar_data():
    states = df['State'].tolist()
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
    tmp_df = df.loc[:, ["Start_Time", "ID"]]
    tmp_df['Start_Time'] = pd.to_datetime(tmp_df['Start_Time'], errors='coerce')
    # column_names = df.columns.values
    # column_names[1] = 'Changed'
    # df.columns = column_names
    # return(tmp_df.groupby([tmp_df['Start_Time'].dt.year, tmp_df['Start_Time'].dt.month]).agg({'count'}))
    return tmp_df.resample('M', on="Start_Time").agg({'count'})


def get_state_frequency():
    tmp_df = df["State"].value_counts().rename_axis('state').reset_index(name='value')
    tmp_df['state'] = tmp_df['state'].map(us_state_abbrev)
    return tmp_df.to_json(orient="records")


def get_pcp_data():
    attributes = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Weather_Condition', 'Wind_Speed(mph)']
    df_tmp = df.sample(500)
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
    print()
    print(df_ts.index.strftime("%Y-%m").tolist())
    message = {}
    message["freq"] = df_ts["Start_Time"]["count"].tolist()
    message["times"] = df_ts.index.strftime("%Y-%m").tolist()

    return message


@app.route("/chloropleth")
def state_frequency():
    myList = get_state_frequency()
    return myList


@app.route("/pcp")
def pcp_data():
    pcp = get_pcp_data()
    # print(pcp)
    return jsonify(pcp)


@app.route("/")
def d3_main():
    return render_template('index1.html')


if __name__ == "__main__":
    app.run(debug=True)
