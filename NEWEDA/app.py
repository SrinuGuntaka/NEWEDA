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
print("Customers Data:")
print(customers.head())
print("\nProducts Data:")
print(products.head())
print("\nTransactions Data:")
print(transactions.head())

# Task 1: Exploratory Data Analysis (EDA)
print("\n--- Exploratory Data Analysis ---")
print("\nCustomers Summary:")
print(customers.info())
print("\nProducts Summary:")
print(products.info())
print("\nTransactions Summary:")
print(transactions.info())

# Missing values check
print("\nMissing Values:")
print("Customers:", customers.isnull().sum())
print("Products:", products.isnull().sum())
print("Transactions:", transactions.isnull().sum())

# Visualizations
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

# Save EDA insights to a PDF
with open("YourName_EDA.pdf", "w") as f:
    f.write("Business Insights:\n")
    f.write("1. Region-wise customer distribution shows XYZ trends.\n")
    f.write("2. The top 5 products contribute significantly to total sales.\n")
    f.write("3. Sales are highest in the ABC region, indicating...\n")
    f.write("4. Transaction volume varies based on product category.\n")
    f.write("5. Signup trends indicate XYZ.\n")

# Task 2: Lookalike Model
print("\n--- Building Lookalike Model ---")
# Merge datasets for a unified view
merged_data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

# Create customer-product interaction matrix
customer_product_matrix = merged_data.pivot_table(index="CustomerID", columns="ProductID", values="Quantity", fill_value=0)

# Compute cosine similarity
similarity_matrix = cosine_similarity(customer_product_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=customer_product_matrix.index, columns=customer_product_matrix.index)

# Find top 3 similar customers for first 20 customers
lookalike_results = {}
for customer_id in customer_product_matrix.index[:20]:
    similar_customers = similarity_df[customer_id].sort_values(ascending=False).iloc[1:4]
    lookalike_results[customer_id] = similar_customers

# Save lookalike results to a CSV
lookalike_df = pd.DataFrame([
    {"CustomerID": k, "SimilarCustomerID": sim_id, "Score": score}
    for k, v in lookalike_results.items() for sim_id, score in v.items()
])
lookalike_df.to_csv("YourName_Lookalike.csv", index=False)
print("Lookalike Model results saved to 'YourName_Lookalike.csv'.")

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
print(f"Davies-Bouldin Index: {db_index}")

# Save clustering results
clustering_data.to_csv("YourName_Clustering.csv", index=False)
print("Clustering results saved to 'YourName_Clustering.csv'.")

# Visualize clusters
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clustering_data['Cluster'], cmap='viridis')
plt.title("Customer Clusters")
plt.xlabel("Quantity (scaled)")
plt.ylabel("Total Value (scaled)")
plt.show()
