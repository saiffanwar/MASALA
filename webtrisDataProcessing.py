import numpy as np
import pandas as pd
#from govuk_bank_holidays.bank_holidays import BankHolidays
import datetime
import os
import matplotlib.pyplot as plt
class DataProcessing:
    '''
    Class containing all the collection and preprocessing methods. From fetching data to passing it to the model.
    '''
    def __init__(self, siteName = None):
        self.daysOfWeek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.siteName = siteName

    def boolToInt(self, bool):
        if bool:
            return 1
        else:
            return 0

    def fetchData(self):
        if self.siteName==None:
            self.siteName = 'fulldata'
        df = pd.read_csv(os.getcwd()+'/webTRIS/data/'+self.siteName+'.csv')
        # df=df.sample(5000)
        df=df.reset_index(drop=True)
        df=df.dropna(axis=1,how='all')
        df['Date'] = [datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in df['Report Date']]
        df.dropna(subset = ["Total Volume"], inplace=True)
        df.dropna(subset = ["Avg mph"], inplace=True)
        df['Day'] = [d.weekday() for d in df['Date']]

        return df

    def fetchTimeIntervalData(self, df, time_interval):
        # df = self.fetchData()
        intervalData = df.loc[df['Time Interval'] == time_interval]
        return intervalData


    def fetchBankHolidays(self):
        bank_holidays = BankHolidays()

        BHols = []
        for bank_holiday in bank_holidays.get_holidays():
            if bank_holiday['date'] < datetime.date(2017, 1,8):
                BHols.append(bank_holiday['date'])
        return BHols

    def checkBankHol(self, day, bankHols):
        return (day in bankHols)

    def checkWeekend(self, day):
        return ((day.weekday() == 5) or (day.weekday() == 6))

    def checkMonday(self, day):
        return ((day.weekday() == 0))


    def timeBins(self, time, period):
            # Convert from string to datetime object
            time = datetime.datetime.strptime(time, '%H:%M:%S').time()
            # Define timepoints in the day.
            morning = datetime.time(6,0,0)
            noon = datetime.time(12,0,0)
            evening = datetime.time(17,0,0)
            night = datetime.time(22,0,0)

            # for the period we are checking, see if it falls within the timeframe and return corresponding bool.
            if period == 'Morning':
                return (morning <= time < noon)
            if period == 'Afternoon':
                return (noon <= time < evening)
            if period == 'Evening':
                return (evening <= time < night)
            if period == 'Night':
                return ((night <= time) or (time < morning))



    def augment(self, data):
        '''
        Adding new features from the existing featuers in the dataset.
        '''

#        year_bank_holidays = self.fetchBankHolidays()
#        data['Bank Holiday'] = list(map(self.boolToInt, list(map(self.checkBankHol, data['Date'], [year_bank_holidays]*len(data['Date'])))))
        data['Weekend'] = list(map(self.boolToInt, list(map(self.checkWeekend, data['Date']))))
        data['Monday'] = list(map(self.boolToInt, list(map(self.checkMonday, data['Date']))))

        data['Day'] = [x.weekday() for x in data['Date']]
        data['Letters in Day'] = [len(self.daysOfWeek[x.weekday()]) for x in data['Date']]

        data['Morning'] = list(map(self.boolToInt, list(map(lambda p: self.timeBins(p, 'Morning'), data['Time Period Ending']))))
        data['Afternoon'] = list(map(self.boolToInt, list(map(lambda p: self.timeBins(p, 'Afternoon'), data['Time Period Ending']))))
        data['Evening'] = list(map(self.boolToInt, list(map(lambda p: self.timeBins(p, 'Evening'), data['Time Period Ending']))))
        data['Night'] = list(map(self.boolToInt, list(map(lambda p: self.timeBins(p, 'Night'), data['Time Period Ending']))))



        # Change 'Date' to ordinal so it can be used in regression
        data['Ordinal Date'] =  np.array([i.toordinal() for i in data['Date']])
        y = data['Total Volume']
        x = data.drop(['Date'], axis=1)
        self.allData = data
        return x, y, data


    def filter(self, x, features=None, normalise=False):
        '''
        Using a list of features we want to use to train the model, remove the other features. If no features are specified, return the full set.
        '''

        if features:
            x = x[features]

        unnormalised = x
        discretes = ['Bank Holiday', 'Weekend', 'Monday', 'Period of Day']
        # Normalise
        if normalise:
            for col in x.columns:
                if col not in discretes:
                    x[col] = [j/np.max(x[col]) for j in x[col]]
        return x


    # def fetchGradient(self, df, feature, gradStep=1):
    #     df['Delta '+feature] = [0 for x in range(len(df))]

    #     for i in tqdm(range(len(df))):
    #         if i >= gradStep:
    #             df['Delta '+feature].iloc[i] = (df[feature].iloc[i] - df[feature].iloc[i-gradStep])/gradStep

    #     return df

    def modelDistribution(self, data):
        fig, axs = plt.subplots(4,4, figsize=(25,15))
        axes = [[i,j] for i in range(4) for j in range(4)]
        for i in range(len(data.columns)):
            axs[axes[i]].hist(data[(data.columns)[i]])




