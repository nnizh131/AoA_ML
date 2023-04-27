import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

cols2 = ['pdda_input_imag1','pdda_input_imag2','pdda_input_imag3','pdda_input_imag4','pdda_input_imag5']
cols1 = ['pdda_input_real1','pdda_input_real2','pdda_input_real3','pdda_input_real4','pdda_input_real5']

def unpack_features(df):
    """Create new column for each IQ value"""
    def unpack(row):
        a,b,c,d,e = row
        return a,b,c,d,e

    df[cols1] = df.apply(lambda row: unpack(row['pdda_input_real']), axis=1, result_type='expand')
    df[cols2] = df.apply(lambda row: unpack(row['pdda_input_imag']), axis=1, result_type='expand')

    return df

def scale(df):
    """scale dataset with mean=0, std=1"""
    scaler = StandardScaler()
    df[cols2] = scaler.fit_transform(df[cols2])
    df[cols1] = scaler.fit_transform(df[cols1])
    return df



def iq_processing(data):
    
    """
    Input: Data
    Output: Processed Data

    Processing: Power Scaling, IQ shifting
    """

    cols_real = ['pdda_input_real_{}'.format(x+1) for x in range(5)]
    cols_imag = ['pdda_input_imag_{}'.format(x+1) for x in range(5)]

    iq_values = pd.DataFrame(data['pdda_input_real'].tolist(), columns=cols_real, index=data.index)
    iq_values[cols_imag] = pd.DataFrame(data['pdda_input_imag'].tolist(), columns=cols_imag, index=data.index)
    
    phase = pd.DataFrame(np.arctan2(iq_values['pdda_input_imag_1'],iq_values['pdda_input_real_1']), columns=['phase_1'])
    
    cos = np.cos(phase).values.ravel()
    sin = np.sin(phase).values.ravel()
    
    out = data.copy()
    iq_ref = np.abs(iq_values[f'pdda_input_real_1']*cos + iq_values[f'pdda_input_imag_1']*sin)
    for i in range(1,6):
        out[f'pdda_input_real_{i}'] = (iq_values[f'pdda_input_real_{i}']*cos + iq_values[f'pdda_input_imag_{i}']*sin)
        out[f'pdda_input_imag_{i}'] = (-iq_values[f'pdda_input_real_{i}']*sin + iq_values[f'pdda_input_imag_{i}']*cos)
        iq_ref +=  iq_values[f'pdda_input_real_{i}']**2 + iq_values[f'pdda_input_imag_{i}']**2

    power_norm =  StandardScaler().fit_transform((out['reference_power'] + out['relative_power']).values.reshape(-1,1))/10
    
    out.insert(25, 'power', power_norm)
    out.insert(24, 'iq_ref', iq_ref)
    out.drop(columns=['pdda_input_imag_1', 'pdda_input_real', 'pdda_input_imag'], inplace=True)
    return out


if __name__ == "__main__":
    df = pd.read_pickle("raw_IQ.pkl")
    df = unpack_features(df)
    df = scale(df)
    df.to_pickle('df.pkl')



