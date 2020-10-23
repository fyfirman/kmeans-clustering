import csv
import random
import math
import matplotlib.pyplot as plt
import numpy as np

"""
Ambil data dan parsing data dari file CSV
"""
def get_data_from_csv(filename):
  with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    result = []
    for data in csv_reader:
      parsed_data = []
      for value in data:
        if(value[0:3]=="ï»¿"):
          parsed_data.append(int(value.strip("ï»¿")))
        else:
          parsed_data.append(int(value))
      result.append(parsed_data)
    return result

"""
Generate random centroid
"""
def get_random_centroids(data, number_of_centroid):
  centroids = []
  while len(centroids) != number_of_centroid:
    random_data = data[random.randrange(0,len(data)-1)]
    if(random_data not in centroids):
      centroids.append(random_data)
  return centroids

"""
Menghitung euclidean distance
"""
def euclidean_distance(point_a, point_b):
  return math.sqrt(pow(point_a[0]-point_b[0],2)+pow(point_a[1]-point_b[1],2))    

"""
Cari tahu titik `point` merupakan anggota cluster mana
"""
def find_members(point, centroids):
  # Initial value
  closest_centroid_index = 0
  min_distance = euclidean_distance(point, centroids[0])

  # Loop and change index & distance if < previous distance
  for centroid_index in range(len(centroids)):
    distance = euclidean_distance(point, centroids[centroid_index])
    if(distance < min_distance):
      min_distance = distance
      closest_centroid_index = centroid_index
  
  return closest_centroid_index + 1

"""
Generate semua point anggota cluster mana
"""
def find_all_members(data, centroids, n_cluster):
    # Buat dictionary kosong untuk menyimpan anggota cluster
    cluster = {}
    for i in range(0,n_cluster):
      cluster[i+1] = []

    for point in data:
      cluster[find_members(point, centroids)].append(point)

    return cluster

"""
Menghitung SSE
"""
def find_sse(cluster, centroids):
  result = 0
  for cluster_index in cluster:
    for point in cluster[cluster_index]:
      result += euclidean_distance(point, centroids[cluster_index-1]) ** 2

  return result

"""
Cari centroid baru (average dari anggota cluster)
"""
def find_new_centroid(cluster, data):
  new_centroids = []
  for cluster_index in cluster:
    if len(cluster[cluster_index]) != 0:
      sumX = 0
      sumY = 0
      for point in cluster[cluster_index]:
        sumX += point[0]
        sumY += point[1]
      new_centroid = [sumX/len(cluster[cluster_index]),sumY/len(cluster[cluster_index])]
      new_centroids.append(new_centroid)

  while len(new_centroids) != len(cluster):
    random_data = data[random.randrange(0,len(data)-1)]
    if(random_data not in new_centroids):
      new_centroids.append(random_data)
  return new_centroids

"""
Menampilkan plot elbow method
"""
def show_sse_elbow_graph(result):
  garis = [[],[]]
  for k in range(len(result)):
    garis[0].append(result[k]['k'])
    garis[1].append(result[k]['sse'])

  plot1 = plt.figure(1)
  plt.axis([garis[0][0],garis[0][len(garis[0])-1],0,garis[1][0]])
  plt.title('Elbow Method') 
  plt.xlabel('Nilai K') 
  plt.ylabel('SSE')
  plt.xticks(range(1,len(garis[0])+1))
  plt.plot(garis[0],garis[1])


"""
Menampilkan plot hasil clustering
"""
def show_clustering_graph(result, k):
  data_raw = []

  # parse data
  for cluster in result[k-1]['cluster']:
    data_raw.append(result[k-1]['cluster'][cluster])

  # put data on plot
  plot2 = plt.figure(2)

  for cluster in data_raw:
    array = np.array(cluster)
    x, y = array.T
    plt.scatter(x,y)

  # put centroid on plot
  centroid_array = np.array(result[k-1]['centroids'])
  x, y = centroid_array.T
  plt.scatter(x,y, color='black', s=50, marker='x')
  plt.show()

if __name__ == "__main__":
  data = get_data_from_csv("data.csv")
  
  result = []

  # Loop dari k=1-20 
  for n_cluster in range(1, 21):
    sse = None
    iterate_count = 0

    # Menentukan centroid secara random
    centroids = get_random_centroids(data,n_cluster)
    
    # Lakukan iterasi hingga sse tidak berubah
    while True:
      cluster = find_all_members(data, centroids, n_cluster)
      
      # pengecekan sse berubah atau tidak
      if sse == None:
        sse = find_sse(cluster, centroids)
      else:
        new_sse = find_sse(cluster, centroids)
        if sse == new_sse:
          result.append({'k': n_cluster, 'sse': new_sse, 'cluster': cluster, 'iterate_count': iterate_count, 'centroids': centroids})
          print({'k': n_cluster, 'sse': new_sse, 'iterate_count': iterate_count})
          break
        else:
          sse = new_sse

      # cari centroid baru
      centroids = find_new_centroid(cluster, data)
      iterate_count += 1

  print(result)

  # menampilkan grafik
  show_sse_elbow_graph(result)
  show_clustering_graph(result,5)