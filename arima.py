import pandas as pd
import numpy
#import 
from numpy import nan
import matplotlib
from matplotlib import pyplot

df=pd.read_csv(r"C:\Users\Avish Khosla\OneDrive\Desktop\smart_grid_sop\household_power_consumption.txt",sep=';',header=0,low_memory=False,infer_datetime_format=True, parse_dates={'datetime':[0,1]},index_col=['datetime'])
print(df.head())
df.replace('?',nan,inplace=True)
val=df.values.astype('float32')
df['sub4']=(val[:,0]*1000/60)-(val[:,4]+val[:,5]+val[:,6])
print(df.head())


# overall plot

pyplot.figure()
for i in range(len(df.columns)):
    pyplot.subplot(len(df.columns),1,i+1)
    name=df.columns[i]
    pyplot.plot(df[name].astype('float32'))
    pyplot.title(name,y=0)
pyplot.show()    


# yearly plot

pyplot.figure()

years=['2007','2008','2009','2010']
for i in range(len(years)):
    ax=pyplot.subplot(len(years),1,i+1)
    year=years[i]
    result=df[str(year)]
    pyplot.plot(result['Global_active_power'].astype('float32'))
    pyplot.title(str(year),y=0,loc='left')
pyplot.show()    


#monthly
pyplot.figure()

years=['2007','2008','2009','2010']
for i in range(1,12):
    ax=pyplot.subplot(12,1,i+1)
    
    result=df['2007'+'-'+str(i)]
    pyplot.plot(result['Global_active_power'].astype('float32'))
    pyplot.title(i,y=0,loc='left')
pyplot.show()    

#histogram plot of variables

pyplot.figure()
for i in range (len(df.columns)):
    pyplot.subplot(len(df.columns),1,i+1)
    result=df[df.columns[i]].astype('float32')
    result.hist(bins=100)
    pyplot.title(df.columns[i],y=0)
pyplot.show()    


# histogram yearwise plots for active power

pyplot.figure()
years=['2007','2008','2009','2010']
for i in range(len(years)):
    pyplot.subplot(len(years),1,i+1)
    result=df[str(years[i])]
    result['Global_active_power'].astype('float32').hist(bins=100)
    pyplot.title(years[i],y=0,loc='left')

pyplot.show()     

