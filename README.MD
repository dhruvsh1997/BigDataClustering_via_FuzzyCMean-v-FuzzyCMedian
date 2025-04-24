# 🔍 Fuzzy Clustering with Spark and PySpark: Analyzing Weblog Data

[![Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=for-the-badge&logo=apache&logoColor=white)](https://spark.apache.org/docs/latest/api/python/index.html)

> A distributed computing approach to analyzing weblog data through fuzzy clustering techniques and machine learning algorithms

![Clustering Banner](https://via.placeholder.com/1200x200?text=Fuzzy+Clustering+with+Spark)

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Introduction](#-introduction)
- [Dataset](#-dataset)
- [Data Preprocessing](#-data-preprocessing)
- [Fuzzy C-Means Clustering](#-fuzzy-c-means-clustering)
- [Linear Regression](#-linear-regression)
- [K-Means Clustering](#-k-means-clustering)
- [Fuzzy C-Median Clustering](#-fuzzy-c-median-clustering)
- [Conclusion](#-conclusion)
- [Future Work](#-future-work)
- [References](#-references)

## 🔭 Project Overview

This Jupyter notebook demonstrates the application of fuzzy clustering techniques, specifically **Fuzzy C-Means (FCM)**, on weblog data using Python and PySpark. The project leverages Spark's distributed computing capabilities to handle large datasets efficiently, making it suitable for Big Data applications.

Key components include:
- Exploratory data analysis of weblog entries
- Linear regression modeling using Spark's MLlib
- K-Means clustering implementation for comparison
- An experimental Fuzzy C-Median clustering approach
- Comprehensive visualization of results

The primary focus is on clustering weblog entries based on IP addresses and status codes to uncover patterns or anomalies that might be relevant for network security and web traffic analysis.

## 🌟 Introduction

**Clustering** is a powerful unsupervised learning technique for uncovering hidden patterns in large datasets, especially in Big Data contexts where scalability is crucial. This project applies **Fuzzy C-Means (FCM)** clustering to a weblog dataset, allowing data points to belong to multiple clusters with varying degrees of membership.

**Why Fuzzy Clustering?**
- Web server logs often contain ambiguous patterns
- User behavior doesn't always fit neatly into distinct categories
- Membership degrees provide more nuanced insights than hard clustering

The project harnesses **PySpark's** distributed computing framework to ensure scalability with growing datasets. Additional analyses including linear regression and K-Means clustering provide a comprehensive exploration and comparison framework.

## 📊 Dataset

The dataset (`weblog.csv`) contains web server logs with fields including:

| Field | Description | Example |
|-------|-------------|---------|
| IP Address | Source IP of the request | "192.168.1.1" |
| Status | HTTP status code | 200 (OK), 404 (Not Found) |

The analysis focuses primarily on these two columns to detect patterns in web traffic and potential anomalies or security issues.

## 🧹 Data Preprocessing

### Loading Data

```python
# Load into PySpark DataFrame for distributed processing
df_pyspark = spark.read.csv('weblog.csv', header=True, inferSchema=True)

# Load into Pandas for easier data manipulation
df = pd.read_csv("weblog.csv")
```

### Cleaning IP Addresses

```python
# Remove dots and convert to integer for numerical analysis
df.IP = df.IP.str.replace('.', '')
df.IP = df.IP.astype('int32')
```

### Handling Missing Values

```python
# Drop rows with missing values
df = df.dropna()
```

### Data Transformation for Spark

```python
# Convert back to Spark DataFrame for distributed processing
sparkDF = spark.createDataFrame(df)
```

## 🧩 Fuzzy C-Means Clustering

**Fuzzy C-Means (FCM)** assigns each data point a membership value for each cluster, allowing for overlapping clusters—particularly useful for weblog data where user behaviors may not fit neatly into distinct groups.

### Data Preparation

```python
# Select relevant columns and convert to NumPy array
dfps = np.array(sparkDF.select("IP", "Staus").collect())
```

### Clustering Process

```python
from fcmeans import FCM

# Test different cluster counts
n_clusters_list = [2, 3, 4, 5, 6, 7]
models = []

for nc in n_clusters_list:
    my_model = FCM(n_clusters=nc)
    my_model.fit(dfps)
    models.append(my_model)
    print(my_model.centers)
```

### Visualization

![FCM Visualization](https://via.placeholder.com/600x300?text=FCM+Clustering+Visualization)

```python
f, axes = plt.subplots(1, 2, figsize=(11,5))
axes[0].scatter(dfps[:,0], dfps[:,1], alpha=.1)
axes[1].scatter(dfps[:,0], dfps[:,1], c=labels, alpha=.1)
axes[1].scatter(centers[:,0], centers[:,1], marker="+", s=500, c='w')
plt.show()
```

### Results

| Number of Clusters | Partition Coefficient (PC) | Partition Entropy Coefficient (PEC) |
|--------------------|----------------------------|------------------------------------|
| 2                  | ~0.5                       | Varies                             |
| 3                  | Lower than 2 clusters      | Varies                             |
| 4                  | Lower than 3 clusters      | Varies                             |
| 5                  | Lower than 4 clusters      | Varies                             |
| 6                  | Lower than 5 clusters      | Varies                             |
| 7                  | Lower than 6 clusters      | Varies                             |

For 2 clusters, PC is approximately 0.5, indicating moderate fuzziness. Higher cluster counts reduce PC, suggesting more distinct clusters.

## 📈 Linear Regression

Linear regression explores the relationship between IP addresses (after conversion) and status codes using Spark MLlib.

### Feature Engineering

```python
from pyspark.ml.feature import VectorAssembler

# Create feature vector
featureassembler = VectorAssembler(inputCols=["IP"], outputCol="IP_ad")
output = featureassembler.transform(sparkDF)
finalized_data = output.select("IP_ad", "Staus")
```

### Model Training

```python
from pyspark.ml.regression import LinearRegression

# Split data
train_data, test_data = finalized_data.randomSplit([0.75, 0.25])

# Train model
regressor = LinearRegression(featuresCol='IP_ad', labelCol='Staus')
regressor = regressor.fit(train_data)
```

### Evaluation

```python
# Evaluate on test data
pred_results = regressor.evaluate(test_data)
pred_results.predictions.show()
```

## 🔷 K-Means Clustering

K-Means is a hard clustering algorithm where each data point belongs to exactly one cluster. Spark MLlib's implementation ensures scalability.

### Standardization

```python
from pyspark.ml.feature import StandardScaler

# Standardize features
scale = StandardScaler(inputCol='IP_ad', outputCol='standardized')
data_scale = scale.fit(finalized_data)
data_scale_output = data_scale.transform(finalized_data)
```

### Clustering Process

```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Evaluate different cluster counts
silhouette_score = []
evaluator = ClusteringEvaluator(predictionCol='prediction', 
                               featuresCol='standardized', 
                               metricName='silhouette', 
                               distanceMeasure='squaredEuclidean')

for i in range(2, 10):
    KMeans_algo = KMeans(featuresCol='standardized', k=i)
    KMeans_fit = KMeans_algo.fit(data_scale_output)
    output = KMeans_fit.transform(data_scale_output)
    score = evaluator.evaluate(output)
    silhouette_score.append(score)
    print("Silhouette Score:", score)
```

### Visualization

![KMeans Silhouette Scores](https://via.placeholder.com/600x300?text=KMeans+Silhouette+Scores)

```python
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.plot(range(2,10), silhouette_score)
ax.set_xlabel('k')
ax.set_ylabel('cost')
plt.show()
```

## 🧮 Fuzzy C-Median Clustering

> **Note**: This section is experimental and uses a non-standard library (fcmid)

The code attempts to implement Fuzzy C-Median clustering, which may use medians instead of means for cluster centers:

```python
import fcmid
fcmedian = fcmid.FCMid(n_clusters=2)
fcmedian.fit(dfps)
centers2 = fcmedian.centers
labels2 = fcmedian.predict(dfps)
```

This approach requires further development or verification of the library.

## 🎯 Conclusion

This project successfully demonstrates the application of fuzzy clustering techniques on weblog data using Python and PySpark, complemented by linear regression and K-Means clustering. Key findings include:

- **Fuzzy C-Means** effectively handles overlapping clusters, ideal for ambiguous weblog data
- **Spark's distributed computing** enables scalable processing for large datasets
- **Complementary insights** from linear regression and K-Means provide a comprehensive analysis
- The **Fuzzy C-Median attempt** highlights opportunities for advanced clustering implementations

## 🔮 Future Work

- Implement FCM directly on Spark for true Big Data scalability
- Develop or source a reliable Fuzzy C-Median algorithm
- Apply these techniques to other datasets to validate their effectiveness
- Explore real-time clustering of streaming weblog data
- Enhance visualization with interactive dashboards

## 📚 References

- [Fuzzy C-Means Clustering](https://en.wikipedia.org/wiki/Fuzzy_clustering)
- [fcmeans Python Library](https://github.com/omadson/fuzzy-c-means)
- [Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [Fuzzy C-Means Clustering Algorithm](https://ieeexplore.ieee.org/document/4767370)

---

<p align="center">
  <i>Discover hidden patterns in your web traffic data with advanced clustering techniques</i>
</p>