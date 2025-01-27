# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Inspect datasets
print("\n--- Customers Data ---")
print(customers.head())

print("\n--- Products Data ---")
print(products.head())

print("\n--- Transactions Data ---")
print(transactions.head())

# Task 1: Exploratory Data Analysis (EDA)
print("\n--- Exploratory Data Analysis ---")
print("\nCustomers Summary:")
print(customers.info())
print("\nProducts Summary:")
print(products.info())
print("\nTransactions Summary:")
print(transactions.info())

# Missing values
print("\nMissing Values:")
print("Customers:\n", customers.isnull().sum())
print("Products:\n", products.isnull().sum())
print("Transactions:\n", transactions.isnull().sum())

# Visualizations
print("\n--- Visualizing Customers by Region ---")
sns.countplot(data=customers, x='Region')
plt.title("Number of Customers by Region")
plt.show()

# Top 5 products by sales
top_products = transactions.groupby('ProductID')['Quantity'].sum().sort_values(ascending=False).head(5)
print("\nTop 5 Products by Sales:")
print(top_products)

# Total sales by region
sales_by_region = transactions.merge(customers, on="CustomerID").groupby('Region')['TotalValue'].sum()
print("\nTotal Sales by Region:")
print(sales_by_region)

# Task 2: Lookalike Model
print("\n--- Building Lookalike Model ---")
# Merge datasets
merged_data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

# Create customer-product interaction matrix
customer_product_matrix = merged_data.pivot_table(index="CustomerID", columns="ProductID", values="Quantity", fill_value=0)

# Compute cosine similarity
similarity_matrix = cosine_similarity(customer_product_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=customer_product_matrix.index, columns=customer_product_matrix.index)

# Find top 3 similar customers for first 20 customers
print("\nTop 3 Similar Customers for Each Customer (First 20):")
for customer_id in customer_product_matrix.index[:20]:
    similar_customers = similarity_df[customer_id].sort_values(ascending=False).iloc[1:4]
    print(f"Customer {customer_id}:")
    print(similar_customers)

# Task 3: Customer Segmentation / Clustering
print("\n--- Customer Segmentation ---")
# Prepare data for clustering
clustering_data = merged_data.groupby("CustomerID").agg({
    'Quantity': 'sum',
    'TotalValue': 'sum'
}).reset_index()

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data.iloc[:, 1:])

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clustering_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Evaluate clustering
db_index = davies_bouldin_score(scaled_data, clustering_data['Cluster'])
print(f"\nDavies-Bouldin Index: {db_index}")

# Display clustering results
print("\nClustering Results (First 10 Customers):")
print(clustering_data.head(10))

# Visualize clusters
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clustering_data['Cluster'], cmap='viridis')
plt.title("Customer Clusters")
plt.xlabel("Quantity (scaled)")
plt.ylabel("Total Value (scaled)")
plt.show()
