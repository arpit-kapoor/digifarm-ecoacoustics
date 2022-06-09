import pandas as pd
import numpy as np
import datetime as dt

class PreProcessing:
    def __init__(self,data_name,metadata_name,data_type):
        self.instrument_data = pd.read_csv(data_name)
        self.instrument_metadata = pd.read_csv(metadata_name)
        self.names = np.unique(self.instrument_data['name'])
        self.formatted_data = self.format_data(data_type)
        self.current_data = self.formatted_data
        self.get_metadata()
        
    def get_metadata(self):
        metadata = self.instrument_metadata.set_index('instrument_name').sort_index()
        self.current_metadata = metadata.loc[self.names]
        return self.current_metadata
        
    def get_names(self):
        return list(self.current_data.columns)
    
    def get_single_frame_data(self,data,index):        
        data = data.iloc[index]
        lats,longs = [],[]
        for name in self.names:
            lats.append(self.current_metadata['latitude'][name])
            longs.append(self.current_metadata['longitude'][name])

        single_frame_data = pd.DataFrame({'long':longs,'lat':lats,'val':data.values})
        
        return single_frame_data
    
    def get_data_for_full_gp(self,data,av_each_date = True):
        
        lats, longs, times,vals = [],[],[],[]
        
        for name in data.columns:
            for time in data[name].index:
                lats.append(self.current_metadata['latitude'][name])
                longs.append(self.current_metadata['longitude'][name])
                times.append(time)
                vals.append(data[name][time])
                
        data_for_gp = pd.DataFrame({'long':longs,'lat':lats,'time':times,'val':vals})
        
        if av_each_date:
            data_for_gp['time'] = data_for_gp.time.apply(lambda x: dt.datetime.timestamp(dt.datetime(x.year,x.month,x.day)))
        else:
            data_for_gp['time'] = data_for_gp.time.apply(lambda x: dt.datetime.timestamp(dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')))

        return data_for_gp
    
    def get_single_site_data(self,data,name):
        return data[name]
        
    def format_data(self,data_type):
        diversity_index = self.instrument_data[self.instrument_data['type']==data_type]
        self.all_times = np.unique(diversity_index['timeStart'])
        
        grouped = diversity_index.groupby(['timeStart','name'], as_index=False).sum()
        grouped = grouped.drop('instrument_id', axis=1)
        pivoted = grouped.pivot(index="timeStart", columns="name").reset_index(drop=True)
        formatted_data = pivoted.droplevel(level=0, axis=1)
        formatted_data.columns.name = None
        formatted_data.fillna(0, inplace=True)

        return formatted_data
    
    def remove_plot(self):
#         This could be improved upon by looking at the intersections
        max_num_zeros = 0
        name_max_num_zeros = ''
        for name in self.names:
            num_zeros = (self.current_data[name]==0).sum()
            if num_zeros > max_num_zeros:
                max_num_zeros = num_zeros
                name_max_num_zeros = name
        self.current_data = self.current_data.drop(columns = [name_max_num_zeros])
        self.names = self.get_names()
        
    def get_sections(self,current_data,name):
        bool_zeros = current_data[:][name].values==0
        toggle = 2
        toggle_array=[]
        non_zero_sections = []
        zero_sections = []
        for i in range(bool_zeros.size):
            if toggle!=bool_zeros[i]:
                toggle = bool_zeros[i]
                toggle_array.append(i)
                if toggle and len(toggle_array)>1:
                    non_zero_sections.append([toggle_array[-2],i])
                elif not toggle and len(toggle_array)>2:
                    zero_sections.append([toggle_array[-2],toggle_array[-1]])
            if i == bool_zeros.size-1:
                if toggle:
                    zero_sections.append([toggle_array[-1],i])
                else:
                    non_zero_sections.append([toggle_array[-1],i])

        return non_zero_sections,zero_sections

    def get_overlapping_intersections(self,current_data):
        all_non_zero_sections = []
        for name in self.names:
            _,zero_sections = self.get_sections(current_data,name)
            for i in range(len(zero_sections)):
                len_section = zero_sections[i][1]-zero_sections[i][0] + 1
                if len_section < 20:
                    self.current_data[name][zero_sections[i][0]:zero_sections[i][1]] = np.nan

            non_zero_sections,_ = self.get_sections(current_data,name)
            all_non_zero_sections.append(non_zero_sections)           
        intersections = self.get_intersections(all_non_zero_sections)
        return intersections

    def get_intersection(self,range_1,range_2):
        x = range(range_1[0],range_1[1])
        y = range(range_2[0],range_2[1])
        xs = set(x)
        inter = xs.intersection(y)
        if len(inter) == 0:
            return []
        return [min(list(inter)),max(list(inter))+1]

    def get_intersections(self,all_non_zero_sections):
        final_set = []
        for i in range(len(all_non_zero_sections[0])):
            base_set = [all_non_zero_sections[0][i]]
            if len(all_non_zero_sections)>1:
                for j in range(1,len(all_non_zero_sections)):
                    new_set = []
                    # Loop over values in base_set
                    for k in range(len(base_set)):
                        # Loop over values in each section
                        for l in range(len(all_non_zero_sections[j])):
                            inter = self.get_intersection(base_set[k],all_non_zero_sections[j][l])
                            if len(inter) != 0:
                                new_set.append(inter)
                    base_set = new_set
            for b_set in base_set:
                final_set.append(b_set)
        return final_set
    
    def get_current_cleaned_data(self):
        ranges = self.get_overlapping_intersections(self.current_data)
        diff_r = []
        for r in ranges:
            diff_r.append(r[1]-r[0])
        max_range_idx = np.argmax(diff_r)
        cleaned_data = self.current_data.iloc[ranges[max_range_idx][0]:ranges[max_range_idx][1]]
        cleaned_and_interpolated_data = self.interpolate_data(cleaned_data)
        self.current_cleaned_data = cleaned_data
        
        cleaned_data['time'] = self.all_times[ranges[max_range_idx][0]:ranges[max_range_idx][1]]
        cleaned_data = cleaned_data.set_index('time')

        return cleaned_data
    
    def interpolate_data(self,data):
        for name in self.names:
            data[name] = data[name].interpolate()
        return data
    
    def get_current_cleaned_date_data(self,df):
        df = df.reset_index()
        df['time'] = df['time'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        df['time'] = df.time.dt.date
        self.current_cleaned_date_data = df.groupby('time', as_index=False).mean().set_index('time')
        return self.current_cleaned_date_data
    
    def get_av_period(self,df):
        df = df.reset_index()
        df['time'] = df['time'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        df['time'] = df.time.dt.date
        df = df.groupby('time', as_index=False).count()
        df = df['A01']
        return int(np.round(np.mean(df.values),0))