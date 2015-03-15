
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 11:42:04 2015

@author: kuttush
"""
#import libraries
import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans, vq, whiten
from pylab import plot,show

un = pd.read_csv('/Users/kuttush/Desktop/Spongebob/Thinkful/DataScience/Unit4/Clustering/un.csv')
print(un.shape) #tells number of rows and columns the database has.
print(un.count()) #returns the total number of non-null values within each column

'''There are 207 rows. The non-null values within each column are: country-207, region-207, tfr-197, contraception-144,
educationMale-76, educationFemale-76, lifeMale-196, lifeFemale-196, infantMortality-201, GDPperCapita-197,
economicActivityMale-165, economicActivityFemale-165, illiteracyMale-160, illiteracyFemale-160.'''

'''country and region are the two columns with the maximum number of non-null values. Thus, we should cluster on these
two columns.'''

list_of_countries = list(set(un['country']))
no_of_country = len(list_of_countries)


#print the type for each column
for i in range(len(un.columns)):
    print(type(un.columns[i]))

'''each of the column is of type str.'''

#extracting the required columns
lifemale = un['lifeMale']
gdp = un['GDPperCapita']
lifefemale = un['lifeFemale']
infantmortality = un['infantMortality']

#replacing the NaN with 0
lifemale[np.isnan(lifemale)]  = 0
gdp[np.isnan(gdp)] = 0
lifefemale[np.isnan(lifefemale)] = 0
infantmortality[np.isnan(infantmortality)] = 0

#Creating a matrix where there are two columsn and 207 rows
d1 = {'gdp': gdp, 'lifemale': lifemale}
df1 = pd.DataFrame(d1)
cluster1 = df1.values
#must be called prior to passing an observation matrix to kmeans. Normalize a group of observations
cluster1 = whiten(cluster1)
centroids1,dist1 = kmeans(cluster1,2)
idx1, idxdist1 = vq(cluster1, centroids1)

'''dist1 = 0.78031267155331685.'''

#plotting
plot(cluster1[idx1==0,0],cluster1[idx1==0,1],'ob',
     cluster1[idx1==1,0],cluster1[idx1==1,1],'or')
plot(centroids1[:,0],centroids1[:,1],'sg',markersize=8)
show()

#now do the same thing for lifefemale and infantmortality
d2 = {'gdp': gdp, 'lifefemale': lifefemale}
df2 = pd.DataFrame(d2)
cluster2 = df2.values
cluster2 = whiten(cluster2)
centroids2, dist2 = kmeans(cluster2,2)
idx2, idxdist2 = vq(cluster2, centroids2)

'''dist2 = 0.79451010274941536'''

plot(cluster2[idx2==0,0],cluster2[idx2==0,1],'ob',
     cluster2[idx2==1,0],cluster2[idx2==1,1],'or')
plot(centroids2[:,0],centroids2[:,1],'sg',markersize=8)
show()

d3 = {'gdp': gdp, 'infantmortality': infantmortality}
df3 = pd.DataFrame(d3)
cluster3 = df3.values
cluster3 = whiten(cluster3)
centroids3, dist3 = kmeans(cluster3,2)
idx3, idxdist3 = vq(cluster3, centroids3)

'''dist3 = 0.85348306502407767.'''

plot(cluster3[idx3==0,0],cluster3[idx3==0,1],'ob',
     cluster3[idx3==1,0],cluster3[idx3==1,1],'or')
plot(centroids3[:,0],centroids3[:,1],'sg',markersize=8)
show()

#clustering with 3 clusters 
d3 = {'gdp': gdp, 'infantmortality': infantmortality}
df3 = pd.DataFrame(d3)
cluster4 = df3.values
cluster4 = whiten(cluster4)
centroids4, dist4 = kmeans(cluster4,3)
idx4, idxdist4 = vq(cluster4, centroids4)

'''dist3 = 0.5346472425089368.'''

plot(cluster4[idx4==0,0],cluster4[idx4==0,1],'ob',
     cluster4[idx4==1,0],cluster4[idx4==1,1],'or',
     cluster4[idx4==2,0],cluster4[idx4==2,1],'og')
plot(centroids4[:,0],centroids4[:,1],'sg',markersize=8)
show()
#clustering with three clusters. 


# can use this one for calculating Euclidian distance
def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])


# check all clusters. Using cluster1 here, but same thing can be done for cluster2 and cluster3
average_distance = []
for k in range(1,11):
    cluster1 = whiten(cluster1)
    centroids1,dist1 = kmeans(cluster1,k) # you can calculate the eucledean distance in the next line
    idx1,idxdist1 = vq(cluster1,centroids1)
    avg_dist = np.mean(idxdist1)
    average_distance.append(avg_dist)
# Just plotting the mean distance, you can plot Euclidian distance once you update the code
plot(range(1,11), average_distance)








'''IGNORE THE CODE BELOW.'''
#creating the data-group and converting each group from list to array because kmeans works only with arrays
#data1 = [lifemale, gdp]
#data1array = np.asarray(data1)
#data2 = [lifefemale, gdp]
#data2array = np.asarray(data2)
#data3 = [infantmortality, gdp]
#data3array = np.asarray(data3)

#apply clustering -- note this is using clustersize=2 because I want to plot. 
#centroids1,dist1 = kmeans(data1array,2)
#idx1,idxdist1 = vq(data1array,centroids1)
#centroids2,dist2 = kmeans(data2array,2)
#idx2,idxdist2 = vq(data2array,centroids2)
#centroids3,dist3 = kmeans(data3array,2)
#idx3,idxdist3 = vq(data3array,centroids3)




'''distance between each point and each cluster centroid and the average within-cluster sum of squares for each centroid are all turning out to be zeros.'''













