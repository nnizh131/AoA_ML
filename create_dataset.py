import pandas as pd
import numpy as np
from collections import defaultdict
import os
import json


rooms = ['testbench_01', 'testbench_01_furniture_low', 'testbench_01_furniture_mid', 'testbench_01_furniture_high']
concrete_rooms = ['testbench_01_furniture_low_concrete', 'testbench_01_furniture_mid_concrete', 'testbench_01_furniture_high_concrete']
other_scenarios = ['testbench_01_rotated_anchors']
anchors = ['anchor1', 'anchor2', 'anchor3', 'anchor4']
channels = ['37','38','39']
polarities = ['V','H']

dataset_name = 'BLE Ray-tracing Simulation Dataset'

def read_files():
    # os.chdir('BLE Ray-tracing Simulation Dataset')
    data = defaultdict(lambda: defaultdict(lambda: defaultdict (lambda: defaultdict(list))))
    anchor_data = defaultdict(lambda: defaultdict(lambda: defaultdict (lambda: defaultdict(list))))

    # create dictionary from json files
    for room in rooms + concrete_rooms + other_scenarios: 
        for channel in channels:  
            for polarity in polarities:   
                tag_filename = f'{dataset_name}/{room}/tag_ml_export_CH{channel}_{polarity}.json'
                tag_df = pd.read_json(tag_filename, orient='records')
                anchor_filename = f'{dataset_name}/{room}/anchor_ml_export_CH{channel}_{polarity}.json'
                anchor_df = pd.read_json(anchor_filename, orient='records')
                df = tag_df.merge(anchor_df)
                # remove calibration points
                df.drop(df[(df['x_tag']==0).values | (df['y_tag']==0).values | (df['z_tag']==0).values].index, inplace=True)
                for anchor in anchors:
                    data[room][anchor][channel][polarity] = df[df['anchor']==int(anchor[-1])]
                    anchor_data[room][anchor][channel][polarity] = anchor_df
    # convert dictionary to dataframe
    df = pd.DataFrame()
    final_df = pd.DataFrame()
    for room in rooms:
        for anchor in anchors:
            for channel in channels:
                for polarity in ['V', 'H']:
                    for k,v in data[room][anchor][channel][polarity].items():
                        temp = pd.DataFrame(v)
                        df = pd.concat([df, v],axis=1)
                    # df['anchor_name'] = anchor
                    df['polarity'] = polarity
                    df['channel'] = channel
                    df['room'] = room
                    final_df = pd.concat([df, final_df],axis=0)
                    df = pd.DataFrame()
    # save df as pickle 
    final_df.to_pickle('raw_IQ.pkl')
    return 


if __name__ == "__main__":
    read_files()


