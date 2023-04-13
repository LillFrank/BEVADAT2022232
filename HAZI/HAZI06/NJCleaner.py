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

        self.data['scheduled_time'] =  pd.to_datetime(self.data['scheduled_time']).dt.hour
        d_map= {pd.Interval(0,3): "late_night", pd.Interval(8,11): "morning", pd.Interval(12, 15) : "afternoon", pd.Interval(16,19):"evening", pd.Interval(20,23):"night", pd.Interval(4,7):"early_morning"}
        self.data['part_of_the_day'] = self.data['scheduled_time'].map(d_map)
        self.data = self.data.drop(['scheduled_time'], axis=1)
        return self.data
    

    def convert_delay(self) -> pd.DataFrame:
        self.data['delay'] =  self.data['delay_minutes'].map(lambda x: 1 if  x > 5 else 0 )
        return self.data
    

    def drop_unnecessary_columns(self) -> pd.DataFrame:
        self.data = self.data.drop(['train_id','scheduled_time' ,'actual_time' , 'delay_minutes'], axis=1 )
        return self.data
    

    def save_first_60k(self, path:str)-> pd.DataFrame:
        df = self.data.head(59999)
        df.to_csv(path,index=False)

    def prep_df(self, path:str)-> None:
        self.data = self.order_by_scheduled_time(self.data)
        self.data = self.drop_columns_and_nan(self.data)
        self.data = self.convert_date_to_day(self.data)
        self.data = self.convert_scheduled_time_to_part_of_the_day(self.data)
        self.data = self.convert_delay(self.data)
        self.data = self.drop_unnecessary_columns(self.data)
        self.save_first_60k(self.data, path)
    

        