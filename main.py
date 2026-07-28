import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create synthetic patient data
# Create a larger synthetic patient dataset
patients = {
    "Patient_ID": list(range(1, 41)),
    "Age_Group": [random.choice(["18-35", "36-55", "56+"]) for _ in range(40)],
    "Fever": [random.choice(["Low", "Medium", "High"]) for _ in range(40)],
    "Cough": [random.choice(["Yes", "No"]) for _ in range(40)],
    "Fatigue": [random.choice(["Yes", "No"]) for _ in range(40)],
    "Housing": [random.choice(["Stable", "Unstable"]) for _ in range(40)],
    "Employment": [random.choice(["Full-time", "Part-time", "Unemployed"]) for _ in range(40)],
    "Education": [random.choice(["High School", "College"]) for _ in range(40)],
    "Social_Support": [random.choice(["Low", "Medium", "High"]) for _ in range(40)],
    "Access_to_Care": [random.choice(["Good", "Poor"]) for _ in range(40)]
}
# Turn the data into a table
patient_data = pd.DataFrame(patients)
# Save the synthetic dataset
patient_data.to_csv("synthetic_patients.csv", index=False)

# Display the dataset
print(patient_data)
# Convert categories into numbers
encoded_data = pd.get_dummies(patient_data)

print(encoded_data)
from sklearn.cluster import KMeans

# Create a K-Means model with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)

# Fit the model to the encoded data
kmeans.fit(encoded_data)

# Save each patient's cluster
patient_data["Cluster"] = kmeans.labels_
# Count how many patients are in each cluster
print("\nNumber of patients in each cluster:")
print(patient_data["Cluster"].value_counts())

# Show the average values for each cluster
print("\nCluster summaries:")
cluster_summary = encoded_data.copy()
cluster_summary["Cluster"] = kmeans.labels_

print(cluster_summary.groupby("Cluster").mean())

# Show the results
print(patient_data)
import matplotlib.pyplot as plt

# Create a scatter plot of the clusters
plt.figure(figsize=(6, 4))
plt.scatter(
    patient_data["Patient_ID"],
    patient_data["Cluster"],
    c=patient_data["Cluster"],
)
plt.title("Patient Clusters")
# Add labels and title
plt.title("Patient Clusters")
plt.xlabel("Patient ID")
plt.ylabel("Cluster")

# Save the graph as an image
plt.savefig("patient_clusters.png", dpi=300)

# Display the graph
plt.show()