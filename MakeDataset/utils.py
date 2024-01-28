import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_date(df, hour=False, drop=True):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    if hour:
        df['hour'] = df['date'].dt.hour
    if drop:
        df.drop('date', axis=1, inplace=True)


def year_month_condition(df, year, month):
    return np.logical_and(df['year'] == year, df['month'] == month)


def print_unique(df):
    lst = []
    for col in df.columns:
        lst.extend(df[col].unique())

    print(f'유니크 값의 개수 : {len(np.unique(lst))}')
    print('유니크 값 목록', np.unique(lst), sep='\n')


def get_crawling(path, mode='Chrome'):
    meta_idx = []
    meta_title = []
    meta_date = []
    meta_use = [65692, 64856, 64381, 46080, 45015, 50742, 53289, 51659, 47120, 35075, 47317, 69103, 78250, 66902, 72249,
                51021, 42588, 44954, 49962, 56998,
                42565, 43681, 51361, 61596, 64874, 59672, 61372, 43513, 45226, 45915, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, 83998, 76603, 52939,
                51642, 56136, 63521, 66437, 55476, 44690, 70294, 81183, 82151, 72449, 65113, 44998, 49435, 52609, 60514,
                70926, 55513, 51356, 64815, 82510,
                62895, 60334, 60334, 55640, 60469, 68710, 75481, 57750, 50299, 60768, 77021, 79959, 74020, 72854, 62632,
                51737, 63271, 61456, 76562, 59113,
                53461, 60613, 70256, 91427, 75979, 65907, 55472, 49558]
    page_number = 9

    driver = webdriver.Chrome()
    driver.get(path)
    time.sleep(1)

    for i in tqdm(range(page_number, 0, -1)):
        driver.find_element(
            By.XPATH, f'//*[@id="menu13422_obj79"]/div[2]/form[3]/div/div/ul/li[{i}]').click()
        time.sleep(1)

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        indexs = soup.select('td.td-num')
        titles = soup.select('td.td-subject')
        dates = soup.select('td.td-date')

        for index in indexs[::-1]:
            meta_idx.append(index.get_text())
        for title in titles[::-1]:
            meta_title.append(title.get_text(strip=True))
        for date in dates[::-1]:
            meta_date.append(date.get_text(strip=True))

    df_meta = pd.DataFrame()
    df_meta['idx'] = meta_idx
    df_meta['title'] = meta_title
    df_meta['date'] = meta_date
    df_meta['using'] = meta_use
    df_meta['title'] = df_meta['title'].str.replace(
        '`', '').str.replace("'", "")
    df_meta['title'] = df_meta['title'].str.split(
        '(', expand=True)[1].str.replace(')', '')

    driver.close()

    # 월 값을 가지는 month 컬럼 생성
    half = df_meta.iloc[:36]
    half['month'] = half['title'].str.replace('월', '').astype(int)

    half2 = df_meta.iloc[36:]
    half2['month'] = half2['title'].str[-3:].str.replace('월', '').astype(int)

    df_meta = pd.concat([half, half2])

    year = 2014
    year__ = []

    for mon in df_meta['month']:
        year__.append(year)
        if mon == 12:
            year += 1

    df_meta['year'] = year__

    # 필요없는 컬럼 제거
    df_meta.drop(['idx', 'title', 'date'], axis=1, inplace=True)

    # 부족한 값 채우기
    df_1719 = df_meta[np.logical_or(df_meta['year'] == 2019, df_meta['year'] == 2017)]
    df_1719 = df_1719.reset_index().drop('index', axis=1)
    df_1719.loc[22] = [np.nan, 2, 2017]
    df_1719.loc[23] = [np.nan, 4, 2019]

    df_else = df_meta[np.logical_and(df_meta['year'] != 2019, df_meta['year'] != 2017)]

    df_meta = pd.concat([df_1719, df_else]).sort_values(['year', 'month'])
    df_meta = df_meta.reset_index().drop('index', axis=1)
    df_meta['month'] = df_meta.pop('month')
    df_meta['using'] = df_meta.pop('using')

    return df_meta