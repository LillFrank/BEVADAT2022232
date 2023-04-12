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


    def convert_scheduled_time_to_part_of_the_day(self)-> pd.DataFrame:
        date_map = {pd.date_range('4:00','8:00'):"early_morning" ,pd.date_range('8:00','12:00'):"morning",pd.date_range('12:00','16:00'):"afternoon",pd.date_range('16:00','20:00'):"evening",pd.date_range('20:00','24:00'):"night" , pd.date_range('0:00','4:00'):"late_night"}
    
        self.data['part_of_the_day'] = self.data['scheduled_time'].map(date_map)
        return self.data
    

    def convert_delay(self) -> pd.DataFrame:
        self.data['delay'] =  self.data['delay_minutes'].map(lambda x: 1 if  x > 5 else 0 )
        return self.data
    

    def drop_unnecessary_columns(self) -> pd.DataFrame:
        self.data = self.data.drop(['train_id','scheduled_time' ,'actual_time' , 'delay_minutes'], axis=1 )
        return self.data
    

    def save_first_60k(self, path_:str)-> pd.DataFrame:
        df = self.data.head(59999)
        df.to_csv(path_,index=False)
    

        