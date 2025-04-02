import csv
import itertools
import matplotlib.pyplot as plt
import pandas as pd

#   increase number of columns displayed when printing
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)

#
#   Pre-Processing
#

data = pd.read_csv(r'housing.csv')


#   Normalization

for column in data:
    if (column == 'ocean_proximity'): continue
    min = data[column].min()
    max = data[column].max()
    data[column] = data[column].apply( lambda x: ( x - min ) / ( max - min ) )

#   Standardization
#   (NOT USED)
'''
for column in data:
    if (column == 'ocean_proximity'): continue

    mean = data[column].mean()
    std = data[column].std()    #   standard deviation
    data[column] = ( data[column] - mean ) / std
print(data.describe()) 
'''

#   One Hot Vector Encoding    
dummies = pd.get_dummies(data.ocean_proximity)
data = pd.concat( [data , dummies], axis='columns' )
data = data.drop( ['ocean_proximity' , 'INLAND' ], axis = 'columns' )


#   Replacing missing values with median
for column in data.columns[data.isnull().any(axis=0)]: # for every column that has null value
    data[column] = data[column].replace( r'^\s+$', data[column].median(), regex=True)
    data[column].fillna( data[column].median() , inplace=True)

#   Exporting the data
data.to_csv("HousingScaled.csv",index=False)



#
#   Data Opticalization
#


#   Frequency Histogram function
def frequency_histogram( figure,  data, label_x ) :
    plt.figure(figure)
    plt.hist(data, bins=30, density=True)    
    plt.xlabel(label_x)
    plt.ylabel("Frequency")
    plt.title( "Probability Density Function for " + label_x )
    return plt



#   opticalization for data after being processed
fig_longtitude = frequency_histogram( 1, data.longitude, "Longitude")
fig_latitude = frequency_histogram( 2, data.latitude, "Latitude")
fig_housing_median_age = frequency_histogram( 3, data.housing_median_age, "housing_median_age")
fig_total_rooms = frequency_histogram( 4, data.total_rooms, "total_rooms")
fig_total_bedrooms = frequency_histogram( 5, data.total_bedrooms, "total_bedrooms")
fig_population = frequency_histogram( 6, data.population, "population")
fig_households = frequency_histogram( 7, data.households, "households")
fig_median_income = frequency_histogram( 8, data.median_income, "median_income")
fig_median_house_value = frequency_histogram( 9, data.median_house_value, "median_house_value")
plt.show()
plt.close('all')


#   opticalization for original data 
OriginalData = pd.read_csv(r'housing.csv')
fig_longtitude = frequency_histogram( 1, OriginalData.longitude, "Longitude")
fig_latitude = frequency_histogram( 2, OriginalData.latitude, "Latitude")
fig_housing_median_age = frequency_histogram( 3, OriginalData.housing_median_age, "housing_median_age")
fig_total_rooms = frequency_histogram( 4, OriginalData.total_rooms, "total_rooms")
fig_total_bedrooms = frequency_histogram( 5, OriginalData.total_bedrooms, "total_bedrooms")
fig_population = frequency_histogram( 6, OriginalData.population, "population")
fig_households = frequency_histogram( 7, OriginalData.households, "households")
fig_median_income = frequency_histogram( 8, OriginalData.median_income, "median_income")
fig_median_house_value = frequency_histogram( 9, OriginalData.median_house_value, "median_house_value")


#   opticalizing ocean_proximity
plt.figure(10)
plt.hist(OriginalData.ocean_proximity)    
plt.xlabel("ocean_proximity")
plt.ylabel("Frequency")
plt.title( "Probability Density Function for ocean_proximity" )
plt.show()
plt.close('all')


#
#   Plotting of 2 variables
#

def Diagram_2_variables( data, column_1, column_2):
    data.plot(x=column_1, y=column_2, kind='scatter')
    plt.xlabel(column_1)
    plt.ylabel(column_2)
    plt.title("Plot with " + column_1 + " and " + column_2)


#   Diagram with housing_median_age & population
subset_data = data[ ["housing_median_age", "population"] ]
Diagram_2_variables( subset_data, "housing_median_age", "population" )

#   Diagram with housing_median_age & total_rooms
subset_data = data[ ["housing_median_age", "total_rooms"] ]
Diagram_2_variables( subset_data, "housing_median_age", "total_rooms" )

#   Diagram with housing_median_age & total_bedrooms
subset_data = data[ ["housing_median_age", "total_bedrooms"] ]
Diagram_2_variables( subset_data, "housing_median_age", "total_bedrooms" )

#   Diagram with total_rooms & total_bedrooms
subset_data = data[ ["total_rooms", "total_bedrooms"] ]
Diagram_2_variables( subset_data, "total_rooms", "total_bedrooms" )

#   Diagram with median_income & median_house_value
subset_data = data[ ["median_income", "median_house_value"] ]
Diagram_2_variables( subset_data, "median_income", "median_house_value" )

plt.show()
plt.close('all')


#
#   Plotting of 3 variables
#

def Diagram_3_variables( data, column_1, column_2, column3):
    plt.figure()
    colormap = plt.cm.get_cmap('jet')
    scattermap =  plt.scatter(data[column_1],data[column_2],c=data[column3],cmap=colormap,alpha=1)
    plt.colorbar(scattermap, label=column3 )
    plt.xlabel(column_1)
    plt.ylabel(column_2)
    plt.title("Plot with " + column_1 + ", " + column_2 + " and " + column3 )


#   Plotting longtitude, latitude, median_house_value
Diagram_3_variables( data, "longitude", "latitude", "median_house_value")

#   Plotting longtitude, latitude, median_house_value
Diagram_3_variables( data, "longitude", "latitude", "median_income")



#   Ploting longtitude, latitude, population
plt.figure()
colormap = plt.cm.get_cmap('jet')
scattermap =  plt.scatter(data["longitude"],data["latitude"], alpha=data["population"])
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Plot of population" )
plt.show()




#
#   Plotting of 4 variables
#


#   Plotting longtitude, latitude, median_house_value and population




plt.figure()
colormap = plt.cm.get_cmap('jet')
scattermap =  plt.scatter(data["longitude"],data["latitude"],c=data["median_house_value"],cmap=colormap, s=data["population"])
plt.colorbar(scattermap, label="median_house_value" )
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Plot of Median_house_value and population" )


plt.show()


#
#   Plotting all variables
#

variables_to_print =["housing_median_age","total_rooms","total_bedrooms","median_income","median_house_value"]
pd.plotting.scatter_matrix(data[variables_to_print])
plt.show()
