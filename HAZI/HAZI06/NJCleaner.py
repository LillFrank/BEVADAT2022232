import pandas as pd


class NJCleaner:
    def __init__(self, csv_path:str) -> None:

        self.data = pd.read_csv(csv_path)


    def order_by_scheduled_time(self) -> pd.DataFrame:
        self.data = self.data.sort_values('scheduled_time')
        return self.data
    
    def drop_columns_and_nan(self) -> pd.DataFrame:
        self.data  = self.data.drop(['from', 'to'], axis=1)
        self.data = self.data.dropna(axis=0)
        return self.data


    def convert_date_to_day(self) -> pd.DataFrame:
        
        self.data['date'] =  pd.to_datetime(self.data['date'])
        self.data['day'] =  self.data['date'].dt.day_name()

        self.data = self.data.drop(['date'], axis=1)
        return self.data
        
    

        