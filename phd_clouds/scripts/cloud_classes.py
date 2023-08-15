import os
import pandas as pd
import numpy as np
import datetime

class Intersection_products:
    def __init__(self,
                 path_class,
                 path_radar):
       
        classification_files = {}
        for file in os.listdir(path_class):
            if file.endswith("classification.nc"):
                classification_files[pd.to_datetime(file[:8], format = "%Y%m%d")] = file
       
        var_files = {}
        for file in os.listdir(path_radar):
            if file.endswith("categorize.nc"):
                var_files[pd.to_datetime(file[:8], format = "%Y%m%d")] = file
        
        #mwr_files = {}
        #for file in os.listdir(path_mwr):
        #    if file.endswith(".nc"):
        #        mwr_files[pd.to_datetime(file[:8], format = "%Y%m%d")] = file
                
        classSet = set(classification_files)
        radarSet = set(var_files)
        #mwrSet   = set(mwr_files)

        intersec_products = [] 
        for date in classSet.intersection(radarSet):
            intersec_products.append(date)
    
        self.var_files          = var_files  # dict 
        self.classification_files = classification_files # dict
        self.dates                = sorted(intersec_products)
        
class HMmodel:
    def __init__(self,
                 lwp,
                 h,
                 z,
                 v=8.7,
                 rw=1e6):
        
        self.cloud_lwp = lwp # liquid water path
        self.cloud_z   = h   # cloud thickeness
        self.radar_ze  = z
        self.nu        = v
        self.rho_w     = rw
        self.k_nv      = (v+3)*(v+4)*(v+5)/( v*(v+1)*(v+2) ) 
        self.k_rv      = (v+2)/( (v+3)*(v+4)*(v+5) )**(1./3.)
        
    def get_num(self):
        # droplet concentration in cm^-3
        #print(integrate.trapz(np.sqrt(self.radar_ze), self.cloud_z))
        # ( g m^-2 / ( g m^-3 mm^3 m^-3/2 * m )  )^2 
        # ( m^3/2 mm^-3 )^2
        #  m^3 (10^-3 m)^-6
        #  m^3 10^-18 m^-6 
        # 10-18 m^-3
        # um^-3
        # To convert from um^-3 to cm^-3: 10^12 * (um^-3) = cm^-3
        return 1e12*self.k_nv*( 6*self.cloud_lwp/( np.pi*self.rho_w*np.trapz(np.sqrt(self.radar_ze), self.cloud_z) ) )**2
        #return 1e12*self.k_nv*( 6*self.cloud_lwp/( np.pi*self.rho_w*np.nansum(np.sqrt(self.radar_ze))*30 ) )**2
    
    def get_re(self):
        # (  g m^-3 mm^3 m^-3/2 * m / ( g m^-2)  )^1/3 mm m^-1/3
        # (  mm^3 m^-3/2  )^1/3 mm m^-1/2
        # (  mm m^-1/2  ) mm m^-1/2
        # (  mm^2 m^-1
        # (  10^-6 m^2 m^-1 )
        # ( 10^-6 m ) = um
        #print(  ( np.pi*self.rho_w*np.trapz(np.sqrt(self.radar_ze), self.cloud_z)/(48*self.cloud_lwp) )**(1./3.) * self.radar_ze**(1./6.)  ) 
        return self.k_rv*( np.pi*self.rho_w*np.trapz(np.sqrt(self.radar_ze), self.cloud_z)/(48*self.cloud_lwp) )**(1./3.) * self.radar_ze**(1./6.)
        


class Cloud_filters:
    def __init__(self, df_class, cloud_type):
        
        self.classification = df_class # Dataframe
        self.height         = df_class.columns
        self.cloud          = cloud_type
        

    
    def quantile_nearest_pixel(self, specie):
        ''' description '''
        row_cloud, col_cloud    = np.where(self.classification == self.cloud)
        row_specie, col_specie  = np.where(self.classification == specie)
        intersection            = np.intersect1d(row_cloud, row_specie) # Temporal index with in Dataframe with coexistence of cloud type and especie
                                                                        # Specie could be ice, rain, aerosol etc....
        no_candidate            = -1.0*self.classification.shape[1]
        dz_bellow               = []
        dz_above                = []
        for i_time in intersection: 
            sequence_cloud  = groupSequence( col_cloud[row_cloud == i_time] ) # liquid pixels sequancies  
            sequence_specie = groupSequence( col_specie[row_specie == i_time] ) 
            
            for cloud_thickness in sequence_cloud:
                
                dbl = no_candidate  # delta bellow the cloud
                ibl = np.nan        # pixel bellow the cloud
        
                dab = abs(no_candidate) # delta bellow the cloud
                iab = np.nan            # pixel bellow the cloud
                
                for specie_thickness in sequence_specie:
            
                    dnew = specie_thickness[-1] - cloud_thickness[0] # new delta bellow the cloud
                    if dnew < 0 and dnew > dbl:
                        dbl = dnew
                        ibl = specie_thickness[-1]
                        
                    dnew = specie_thickness[0] - cloud_thickness[-1] # new delta above the cloud
                    if dnew > 0 and dnew < dab:
                        dab = dnew
                        iab = specie_thickness[0]
         
                if not np.isnan(ibl) and not np.isnan(iab):
                    dz_bellow.append( self.height[cloud_thickness[0]] - self.height[ibl] )
                    dz_above.append( self.height[iab] - self.height[cloud_thickness[-1]] )
                elif not np.isnan(ibl):
                    dz_bellow.append( self.height[cloud_thickness[0]] - self.height[ibl] )
                elif not np.isnan(iab):
                    dz_above.append( self.height[iab] - self.height[cloud_thickness[-1]] )
                  
        return dz_bellow, dz_above
        
    def filtering_specie(self, specie, dzb_max, dzt_max):
        ''' description '''
        row_cloud, col_cloud    = np.where(self.classification == self.cloud)
        row_specie, col_specie  = np.where(self.classification == specie)
        intersection            = np.intersect1d(row_cloud, row_specie) # Temporal index with in Dataframe with coexistence of cloud type and especie
        
        if intersection.any():                                                                # Specie could be ice, rain, aerosol etc....
            no_candidate        = -1.0*self.classification.shape[1]
            for i_time in intersection: 
                sequence_cloud  = groupSequence( col_cloud[row_cloud == i_time] ) # liquid pixels sequancies  
                sequence_specie = groupSequence( col_specie[row_specie == i_time] ) 
            
                for cloud_thickness in sequence_cloud:
                    dbl = no_candidate      # delta bellow the cloud
                    dab = abs(no_candidate) # delta bellow the cloud
                    ibl = np.nan            # pixel bellow the cloud
                    iab = np.nan            # pixel bellow the cloud
                
                    for specie_thickness in sequence_specie:
                        dnew = specie_thickness[-1] - cloud_thickness[0] # new delta bellow the cloud
                        if dnew < 0 and dnew > dbl:
                            dbl = dnew
                            ibl = specie_thickness[-1]
                        
                        dnew = specie_thickness[0] - cloud_thickness[-1] # new delta above the cloud
                        if dnew > 0 and dnew < dab:
                            dab = dnew
                            iab = specie_thickness[0]
        
                    if not np.isnan(ibl) and not np.isnan(iab):
                        #print("have ice bellow and above liquid cloud")
                        dzb = self.height[cloud_thickness[0]] - self.height[ibl]
                        dzt = self.height[iab] - self.height[cloud_thickness[-1]]
                        #print(dzb)
                        if dzb <= dzb_max or dzt <= dzt_max:
                            self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan
                        #self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan
                    elif not np.isnan(ibl):
                        dzb = self.height[cloud_thickness[0]] - self.height[ibl]
                        #print(dzb)
                        if dzb <= dzb_max:
                            self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan
                        #self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan
                    elif not np.isnan(iab):
                        dzt = self.height[iab] - self.height[cloud_thickness[-1]]
                        if dzt <= dzt_max:
                            self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan
                        #self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1]+1] = np.nan

    def filtering_dz(self, dz_min):
        """ description """
        #filtered_dz = self.classification.copy()
        row, col        = np.where(self.classification == self.cloud)
        row_unique      = np.unique(row)
        
        if row_unique.any():
            for i_time in row_unique:
                sequence_z = groupSequence( col[row==i_time] )
                for group_z in sequence_z:
                    dz = self.height[group_z].max() - self.height[group_z].min()
                    if dz < dz_min:
                        self.classification.iloc[i_time,group_z] = np.nan

    def filtering_dt(self, dt_min):
        """ description """
        #filtered_dt = self.classification.copy()
        row, col        = np.where(self.classification == self.cloud)
        row_unique      = np.unique(row)
        
        if row_unique.any():
            sequence_time   = groupSequence(row_unique)
            for group_t in sequence_time:
                dt = self.classification.index[group_t].max() - self.classification.index[group_t].min()
                if dt < datetime.timedelta(minutes=dt_min):
                    for i_time in group_t:
                        sequence_z = groupSequence( col[row==i_time] )
                        for group_z in sequence_z:
                            self.classification.iloc[i_time,group_z] = np.nan

    def filtered_ze(self, ze):
        """ description """
        #return self.reflectivity.where(self.classification == self.cloud, np.nan)
        return ze.where(self.classification == self.cloud, np.nan)
        
        #for group_t in sequence_time:
        #    dt = self.classification.index[group_t].max() - self.classification.index[group_t].min()
        #    print(self.classification.index[group_t].min(),
        #          self.classification.index[group_t].max())
        #    #if dt < datetime.timedelta(minutes=m):
        #    for i_time in group_t:
        #        sequence_z = groupSequence( col[row==i_time] )
        #        #n = len(sequence_z)
        #        #if n > 1:
        #        #    print(self.classification.index[i_time])
        #        for group_z in sequence_z:
        #            z  = filtered_limits.columns[group_z]
        #            dz = z[-1] - z[0]
        #            #print(sum(np.diff(sequence_z) > 60))
        #            #print(self.classification.index[i_time], dt < datetime.timedelta(minutes=m))
        #            if dz < dz_min or dt < datetime.timedelta(minutes=m):
        #                #print(dz)
        #                filtered_limits.iloc[i_time,group_z] = np.nan 
        
        #return filtered_limits
        