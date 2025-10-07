import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

cell_size = 220
center_offset = (0, 0)

df = pd.read_csv('datasets/simbarca/metadata/link_bboxes.csv')

df['c_x'] = (df['from_x'] + df['to_x'])/2 + center_offset[0]
df['c_y'] = (df['from_y'] + df['to_y'])/2 + center_offset[1]

# cluster the points according to c_x and c_y
X = df[['c_x', 'c_y']].values
kmeans = KMeans(n_clusters=4, random_state=42).fit(X)
df['cluster'] = kmeans.labels_

# divide all points into grids with cell_size, and assign grid ID to each point (from 0)
df['grid_x'] = df['c_x'] // cell_size
df['grid_y'] = df['c_y'] // cell_size
df['grid_nb'] = df.groupby(['grid_x', 'grid_y']).ngroup()

# plot the clusters along with a grid of length cell_size
plt.figure(figsize=(6, 4.5))
plt.scatter(df['c_x'], df['c_y'], c=df['cluster'], s=2)
plt.grid(which='both', linestyle='--', linewidth=1, color='gray', alpha=0.5)
# Show horizontal grid lines
plt.yticks(ticks=np.arange(df['grid_y'].min()*cell_size, (df['grid_y'].max()+2)*cell_size, cell_size), 
           labels=np.arange(0, df['grid_y'].max() - df['grid_y'].min()+2, 1).astype(int))
# Show vertical grid lines
plt.xticks(ticks=np.arange(df['grid_x'].min()*cell_size, (df['grid_x'].max()+2)*cell_size, cell_size),
           labels=np.arange(0, df['grid_x'].max() - df['grid_x'].min()+2, 1).astype(int), rotation=90)
plt.xlabel('X Grid ID')
plt.ylabel('Y Grid ID')
plt.title('Region clusters and monitoring grids')
plt.tight_layout()
plt.savefig('datasets/simbarca/figures/cluster.pdf')


# save the modified dataframe to a csv file
df.to_csv('datasets/simbarca/metadata/link_bboxes_clustered.csv', index=False)
