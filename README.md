# Mall Customer Segmentation using K-Means Clustering

## Project Overview
This project applies **K-Means Clustering** to segment mall customers based on their purchasing behavior, demographics, and spending habits. By identifying different customer groups, businesses can tailor marketing strategies, enhance customer engagement, and optimize customer relationship management.

## Features
- Segments customers based on spending habits, income, and other factors.
- Visualizes clusters for better understanding of customer behavior.
- Allows businesses to target customer groups with personalized offers.

## Prerequisites
Ensure you have the following installed:
- Python 3.x
- Pandas (`pip install pandas`)
- NumPy (`pip install numpy`)
- Matplotlib (`pip install matplotlib`)
- Seaborn (`pip install seaborn`)
- Scikit-learn (`pip install scikit-learn`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mall-customer-segmentation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd mall-customer-segmentation
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The dataset used is `Mall_Customers.csv`, which contains:
- **CustomerID**: Unique ID for each customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Annual Income (k$)**: Annual income of the customer.
- **Spending Score (1-100)**: Score assigned to the customer based on spending behavior.

## Usage
1. Load the dataset:
   ```python
   import pandas as pd
   df = pd.read_csv('Mall_Customers.csv')
   ```

2. Preprocess the data:
   ```python
   # Select features for clustering
   X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
   ```

3. Perform K-Means Clustering:
   ```python
   from sklearn.cluster import KMeans

   # Set number of clusters
   kmeans = KMeans(n_clusters=5, random_state=0)
   y_kmeans = kmeans.fit_predict(X)
   ```

4. Visualize the clusters:
   ```python
   import matplotlib.pyplot as plt

   plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans, cmap='viridis')
   plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
   plt.title('Mall Customer Segments')
   plt.xlabel('Annual Income (k$)')
   plt.ylabel('Spending Score (1-100)')
   plt.legend()
   plt.show()
   ```

## Visualization
The plot shows customer segments in different colors, with red points representing the cluster centroids. This allows businesses to easily identify different customer groups.

## Key Insights
- Customers are segmented based on income and spending score.
- Each cluster represents customers with similar spending behavior, which can help businesses develop targeted marketing strategies.

## Future Improvements
- Add more features to improve clustering accuracy, such as age or location.
- Perform elbow method or silhouette analysis to find the optimal number of clusters.

## License
This project is licensed under the MIT License.
```

This README covers installation, usage, dataset details, and clustering process, with sample code included for ease of use.
