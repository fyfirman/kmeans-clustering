import kmeans
import math

def test_get_data_from_csv():
  data = [[15, 39], [15, 81], [16, 6]]
  assert kmeans.get_data_from_csv('mock_data.csv') == data, "harus sama"

def test_get_random_centroid():
  data = [[15, 39], [15, 81], [16, 6], [12, 4]]
  result = kmeans.get_random_centroids(data, 2)

  contains = 0
  for value in data:
    if(value in result):
      contains = contains + 1
  
  assert contains == 2, "harus keluar 2"

def test_euclidean_distance():
  point_a = [2,5]
  point_b = [2,5]
  
  assert kmeans.euclidean_distance(point_a, point_b) == 0, "harus keluar 0"

def test_find_members():
  point = [6.5,2.2]
  centroids = [[2,5.5],[5,3.5]]

  assert kmeans.find_members(point, centroids) == 2, "harus keluar 2"

def test_find_all_members():
  points = [[2,5], [2,5.5], [5,3.5], [6.5,2.2], [7,3.3], [3.5,4.8], [4,4.5]]
  centroids = [[2,5.5],[5,3.5]]
  result = {1: [[2, 5], [2, 5.5], [3.5, 4.8]], 2: [[5, 3.5], [6.5, 2.2], [7, 3.3], [4, 4.5]]}

  assert kmeans.find_all_members(points, centroids, 2) == result, "harus sesuai dengan hasil"

def test_find_sse():
  cluster = {1: [[2, 5], [2, 5.5], [3.5, 4.8]], 2: [[5, 3.5], [6.5, 2.2], [7, 3.3], [4, 4.5]]}
  centroids = [[2,5.5],[5,3.5]]

  result = 12.97

  assert math.isclose(kmeans.find_sse(cluster, centroids), result, abs_tol=0.01), "harus sesuai dengan hasil"
  
def test_find_new_centroid():
  data = [[2,5], [2,5.5], [5,3.5], [6.5,2.2], [7,3.3], [3.5,4.8], [4,4.5]]
  cluster = {1: [[2, 5], [2, 5.5], [3.5, 4.8]], 2: [[5, 3.5], [6.5, 2.2], [7, 3.3], [4, 4.5]], 3: []}
  result = [[2.50, 4.27], [5.63, 3.38]]
  
  is_correct = False
  data_test = kmeans.find_new_centroid(cluster, data)
  for i in range(len(result)):
    for j in range(len(result[i])):
      is_correct = math.isclose(data_test[i][j], result[i][j], abs_tol=0.01)
  
  assert is_correct == True, "harus sesuai dengan hasil"


if __name__ == "__main__":
  test_get_data_from_csv()
  test_get_random_centroid()
  test_euclidean_distance()
  test_find_members()
  test_find_all_members()
  test_find_sse()
  test_find_new_centroid()
  print("Everything passed")