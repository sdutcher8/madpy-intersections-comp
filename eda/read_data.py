import pandas as pd
import numpy as np

def read_data(city='Chicago'):
    """
    This function will read in the desired City and create new veriables
    like TurnDegree, ExitStreetId and so forth
    """
    df = pd.read_csv('train.csv')
    df = df[df.City==city]

    # I will be using this dictionary to convert direction to degrees
    degrees = {'N':0, 'NE':45, 'E':90, 'SE':135, 'S':180, 'SW':225, 'W':270, 'NW':315}

    df["EntryHeading_deg"] = df.EntryHeading.apply(lambda x:degrees[x])
    df["ExitHeading_deg"] = df.ExitHeading.apply(lambda x:degrees[x])
    df["TurnDegree"] = (df.EntryHeading_deg-df.ExitHeading_deg).apply(lambda x: x if abs(x) <=180 else (x+360 if x<0 else x-360))
    df["TurnDegre"] = df.TurnDegree.apply(lambda x: x if x != -180 else x*-1)

    # Lets assign a number(StreetId) to each street
    all_streets = np.concatenate([df.ExitStreetName.reindex().values, df.EntryStreetName.reindex().values])
    # there are some nan values so lets just replace them with Unknown
    street_name_list = ['Unknown' if type(x)==type(0.0) else x for x in all_streets]
    street_names = {name: num for num, name in enumerate(street_name_list)}
    df["EntryStreetId"] = np.array([street_names[x] if x in street_names else -999 for x in df.EntryStreetName])
    df["ExitStreetId"] = np.array([street_names[x] if x in street_names else -999 for x in df.ExitStreetName])

    # we also want to categorize the street by its type (road, boulevard, ...)
    street_types = {n: i for i, n in enumerate(np.unique([x.split()[-1] for x in street_names.keys()]))}
    street_name_to_type = {}
    for name in street_names.keys():
        typ = name.split()[-1]
        street_name_to_type[name] = street_types[typ]
    df["EntryStreetType"] = np.array([street_name_to_type[x] if x in street_names else -999 for x in df.EntryStreetName])
    df["ExitStreetType"] = np.array([street_name_to_type[x] if x in street_names else -999 for x in df.ExitStreetName])

    df["EnterHighway"] = np.array([1 if type(x)==type('') and x.split()[-1] in ['Broadway', 'Parkway', 'Expressway', 'Highway'] else 0 for x in df.EntryStreetName])
    df["ExitHighway"] = np.array([1 if type(x)==type('') and x.split()[-1] in ['Broadway', 'Parkway', 'Expressway', 'Highway'] else 0 for x in df.ExitStreetName])
    df['Season'] = np.array([1 if month in (12,1,2) else 2 if month in (6,7,8) else 3 for month in df.Month.reindex().values])
    df['RushHour'] = np.array([1 if hour in (7,8,9) else 2 if hour in (16,17,18) else 3 if hour>=10 and hour<=15 else 4 for hour in df.Hour])
    return df

if __name__ == "__main__":
    df = read_data('Chicago')

    cols = ['Latitude', 'Longitude', 'Season', 'RushHour', 'Weekend',
                  'EntryHeading_deg', 'ExitHeading_deg', 'TurnDegree',
                  'EnterHighway', 'ExitHighway',
                  'TimeFromFirstStop_p80']
    x_train, x_test, y_train, y_test, scaler_x, scaler_y = create_train_test(df, cols)

