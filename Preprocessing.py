#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Import Module
import os
import warnings
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import medfilt

warnings.filterwarnings(action='ignore')
plt.style.use('ggplot')
sns.set_style("darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


def getfiles() -> dict:
    """
    상대경로를 이용해 파일들 가져옴.
    이를 위해서는 같은 파일 내에 /data 폴더가 있어야 하고,
    /data 폴더 내에는 /userOO 형의 폴더가 있어야 함.
    
    Args:
        - 
    Returns:
        dfs(str) : 각 참가자 별 비디오를 저장해놓은 dict.
                   "userOO_video_OO" 형식을 key로 가짐.        
    """
    file_list = []
    dfs = {}
    for dirname,_,filenames in os.walk(os.getcwd()):
        if dirname.find('user') != -1:
            user_name = str(dirname.split('\\')[-1])
            video_cnt = 1
            for file in filenames:
                if file.find('비고') != -1 :
                    continue
                if video_cnt < 10:
                    video_name = '_video_0' + str(video_cnt)
                else:
                     video_name = '_video_' + str(video_cnt)
                tmp = pd.read_csv(dirname + "\\" + file)
                tmp = tmp.fillna(0)
                dfs[user_name + video_name] = tmp
                video_cnt += 1
    return dfs


# In[12]:


def preprocessing_time(df:pd.DataFrame) -> pd.DataFrame:
    """
    맨 앞의 띄어쓰기를 삭제하고,
    Date column을 지우고,
    Time column을 초 단위로 만듦.
    
    Args: 
        df(DataFrame) : 각 참여자의 한 비디오 데이터프레임
    Returns:
        df(DataFrame) : preprocessed된 비디오 데이터프레임
    """
    # 컬럼 이름 바꿈
    df.columns = ['Date', 'Time', 'Left_pos.x', 'Left_pos.y', 'Left_pos.z',
       'Left_rot.x', 'Left_rot.y', 'Left_rot.z', 'EDA', 'BVP', 'TMP',
       'combined_x', 'combined_y', 'combined_z', 'r_bit', 'r_open',
       'r_dir_x', 'r_dir_y', 'r_dir_z', 'r_gaze_origin_x',
       'r_gaze_origin_y', 'r_gaze_origin_z', 'r_pupil', 'r_pupil_pos',
       'r_pupil_pos_x', 'r_pupil_pos_y', 'r_frown', 'r_squeeze',
       'r_widel_bit', 'l_open', 'l_dir_x', 'l_dir_y', 'l_dir_z',
       'l_gaze_origin_x', 'l_gaze_origin_y', 'l_gaze_origin_z', 'l_pupil',
       'l_pupil_pos', 'l_pupil_pos_x', 'l_pupil_pos_y', 'l_frown',
       'l_squeeze', 'l_wide']
    
    # Date column 드랍
    df.drop(['Date'], axis = 1, inplace = True)
    
    # 시간 자료형을 정수형으로 바꿈
    # ex. 15:18:07 을 1로(시작하는 시간), 15:18:53을 46(끝나는 시간)으로 바꿈
    cnt = 1
    for i in range(len(df)-1):

        # 마지막을 제외한 모든 case
        if df['Time'][i] != df['Time'][i+1]:           # 1초가 넘어가면
            df['Time'][i] = cnt
            cnt += 1                                   # cnt 1 증가
        else:
            df['Time'][i] = cnt
    #마지막 element 처리        
    df['Time'].iloc[-1] = cnt
    
    return df


# In[13]:


def column_drop(df:pd.DataFrame) -> pd.DataFrame:
    """
    필요없는 column 드랍
    """
    df = df.drop(['r_pupil_pos_y','r_frown','r_squeeze',
                'l_pupil_pos_y','l_frown','l_squeeze', 'l_wide'], axis = 1)

    df.columns = ['Time', 'Left_pos.x', 'Left_pos.y', 'Left_pos.z',
       'Left_rot.x', 'Left_rot.y', 'Left_rot.z', 'EDA', 'BVP', 'TMP',
       'combined_x', 'combined_y', 'combined_z', 
       'r_bit', 'r_open','r_dir_x', 'r_dir_y', 'r_dir_z', 'r_gaze_origin_x',
       'r_gaze_origin_y', 'r_gaze_origin_z', 'r_pupil', 'r_pupil_pos_x',
       'r_pupil_pos_y',
       'l_bit', 'l_open', 'l_dir_x', 'l_dir_y',
       'l_dir_z', 'l_gaze_origin_x', 'l_gaze_origin_y', 'l_gaze_origin_z',
       'l_pupil', 'l_pupil_pos_x', 'l_pupil_pos_y']
    return df


# In[14]:


def phys_preprocessing(df:pd.DataFrame) -> pd.DataFrame:
    """
    physical data를 preprocessing함.
    필요 없는 문자열 데이터를 없애고, float 형태로 preprocessing함.
    이와 함께, 결측치들을 대체해줌(이후, 마지막에 median filter를 반영해줄 것임)
    
    Args: 
        df(DataFrame) : 각 참여자의 한 비디오 데이터프레임
        
    Returns:
        df(DataFrame) : preprocessed된 비디오 데이터프레임
    """
    ############################ EDA ############################
    # default : 결측치를 의미한다고 생각. EDA 측정 전 대기시간?
    # default를 가장 가까이 있는 데이터로 대체.
    default = ['default', ' default', '', ' ',
               ' R device_connect O', ' R device_subscribe gsr O',
               ' R device_subscribe bvp O', 'R device_subscribe tmp O',
              ' R device_subscribe tmp O', np.nan]
#     print(default)
    df['EDA'] = df['EDA'].apply(lambda x: np.nan if x in default else x)
#     print(df['EDA'])
    df['EDA'] = df['EDA'].apply(lambda x: x.split(' ')[-1] if (x is not np.nan) else x)
    df['EDA'] = pd.to_numeric(df['EDA'], errors='coerce') # float가 아니면 NaN
    df['EDA'] = df['EDA'].interpolate(method = 'nearest')
    near = df['EDA'][max(df[df['EDA'].isna()].index) + 1]
    df['EDA'].replace(to_replace = default, value = near, inplace = True)

    
    ############################ BVP ############################
    # 아무것도 없는 값이 있음. 
    # 이것도 위의 EDA와 똑같은 처리를 해준다.
    
    df['BVP'] = df['BVP'].apply(lambda x: np.nan if x in default else x)
    # 시간 값을 모두 없애고 유의미한 값만 남김, float로 만듦
    df['BVP'] = df['BVP'].apply(lambda x: x.split('.')[-1] if (x is not np.nan) else x)
    df['BVP'] = pd.to_numeric(df['BVP'], errors='coerce') # float가 아니면 NaN
    df['BVP'] = df['BVP'].interpolate(method = 'nearest')
    near = df['BVP'][max(df[df['BVP'].isna()].index) + 1]
    df['BVP'].replace(to_replace = default, value = near, inplace = True)
    
    ############################ TMP ############################
    # 결측값에는 다 O가 들어가 있기 때문에 모두 확인.
    # 이것도 시작하기 전 맨 앞에 들어가있음을 확인할 수 있음.
    df['TMP'] = df['TMP'].apply(lambda x: ' R device_subscribe tmp O' if len(x) < 0 else x)
    def f(x):
        if 'O' in x:
            return x
        else:
            return np.nan
    try:
        ind = df[df['TMP'].apply(f).notnull()].index
        near = df['TMP'][np.max(ind) + 1]
        df['TMP'].replace([' R device_subscribe tmp O',
                       ' R device_subscribe gsr O',
                       ' R device_subscribe bvp O'], near, inplace = True)
    except KeyError: #마지막 값이 0인 경우
            ind = df[df['TMP'].apply(f).notnull()].index
            near = df['TMP'][np.max(ind) - 1]
            df['TMP'].replace([' R device_subscribe tmp O',
                       ' R device_subscribe gsr O',
                       ' R device_subscribe bvp O'], near, inplace = True)
    except ValueError:
        pass
    
    # 시간 값을 모두 없애고 유의미한 값만 남김, float로 만듦
    df['TMP'] = df['TMP'].apply(lambda x: x.split(' ')[-1] if (x is not np.nan) else x)
    df['TMP'] = pd.to_numeric(df['TMP'], errors='coerce') # float가 아니면 NaN
    try:
        near = df['TMP'][df[df['TMP'].isna()].index + 1]
#         near = df['TMP'][np.argmax(df[df['TMP'] == np.nan].index) + 1]
        df['TMP'].replace([np.nan], near, inplace = True)
    except ValueError: # 만약 default 값이 없는 경우
        pass
    
    return df


# In[15]:


def median_filter(df:pd.DataFrame) -> pd.DataFrame:
    """
    median filter를 사용해 모든 센서 데이터 smoothing
    """

    target_col = ['Left_pos.x', 'Left_pos.y', 'Left_pos.z',
       'Left_rot.x', 'Left_rot.y', 'Left_rot.z', 'EDA', 'BVP', 'TMP',
       'combined_x', 'combined_y', 'combined_z', 
       'r_bit', 'r_open','r_dir_x', 'r_dir_y', 'r_dir_z', 'r_gaze_origin_x',
       'r_gaze_origin_y', 'r_gaze_origin_z', 'r_pupil', 'r_pupil_pos_x',
       'r_pupil_pos_y',
       'l_bit', 'l_open', 'l_dir_x', 'l_dir_y',
       'l_dir_z', 'l_gaze_origin_x', 'l_gaze_origin_y', 'l_gaze_origin_z',
       'l_pupil', 'l_pupil_pos_x', 'l_pupil_pos_y']

    df[target_col] = df[target_col].apply(medfilt, kernel_size = 7, axis = 0)

    return df


# In[16]:


def padding_to_numeric(df:pd.DataFrame) -> pd.DataFrame:
    """
    median filter 이후의 padding이 0인 부분을 모두 숫자로 바꿔줌
    
    Args: 
        df(DataFrame) : 각 참여자의 한 비디오 데이터프레임
    Returns:
        df(DataFrame) : preprocessed된 비디오 데이터프레임
    """
    for column in df.columns:
        try:
            near = df[column][np.argmax(df[df[column] == 0].index) + 1]
            df[column].replace([0], near, inplace = True)
        except KeyError: #마지막 값이 0인 경우
            near = df[column][np.argmax(df[df[column] == 0].index) - 1]
            df[column].replace([0], near, inplace = True)
        except ValueError: # 만약 default 값이 없는 경우
            pass
    
    return df


# In[17]:


def write(df:pd.DataFrame, name:str) -> None:
    """
    각 데이터프레임을 파일로 내보냄
    """
    if not os.path.exists(os.path.join(os.getcwd(), "preprocessed_data")):
        os.mkdir(os.path.join(os.getcwd(), "preprocessed_data"))
    path = os.path.join(os.getcwd(), "preprocessed_data")
    
    if not os.path.exists(os.path.join(path, name[:6])):
        os.mkdir(os.path.join(path, name[:6]))
    path2 = os.path.join(path, name[:6])
    
    df.to_csv(os.path.join(path2,name[-2:]) + '.csv', encoding = 'utf-8')
# 
    


# In[18]:


if __name__ == "__main__":
    dfs = getfiles()
    for video in dfs:
        print(video)
        a = dfs[video].copy()
        b = preprocessing_time(a)
        c = column_drop(b)
        d = phys_preprocessing(c)
        e = median_filter(d)
        res = padding_to_numeric(e)
        write(res, video)

