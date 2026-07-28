# Valley Fever Patient Clustering Project

## Overview

This project demonstrates how machine learning techniques can be used to analyze healthcare related data and identify patterns within patient information. Using a synthetic patient dataset, this project applies K-Means clustering to group patients based on similarities in selected characteristics.

## Tools Used

- Python
- pandas
- scikit-learn
- matplotlib
- K-Means clustering

## What This Project Does

This program:
- Creates a synthetic patient dataset representing healthcare-related information.
- Converts categorical patient information into a format that machine learning algorithms can analyze.
- Uses K-Means clustering to automatically group patients with similar characteristics.
- Excludes patient identifiers from the clustering process to prevent irrelevant information from influencing the results.
- Generates cluster summaries to help interpret the characteristics of each patient group.
- Uses Principal Component Analysis (PCA) to visualize the multi-dimensional patient clusters in a two-dimensional graph.
- Produces reproducible results by using a fixed random seed.

## Dataset Features

The synthetic patient dataset includes healthcare-related and social factors used for clustering analysis:

- Patient ID
- Age group
- Fever severity 
- Cough symptoms
- Fatigue
- Housing stability
- Employment status
- Education level
- Social support
- Access to care

These features were used to identify patterns and group patients with similar characteristics using K-Means clustering.

## Project Outcomes
The final visualization demonstrates how machine learning can be used to uncover patterns within patient data. This project provides experience with healthcare data organization, machine learning workflows, and data visualization.

## Project Visualization

Below is a visualization of the patient clusters identified by the K-Means clustering algorithm.

![Patient Clusters](patient_clusters.png)

## Future Improvements

Potential future improvements for this project include:

- Applying the analysis to real-world de-identified healthcare datasets.
- Exploring additional machine learning approaches for patient pattern identification.
- Evaluating different clustering methods and parameters to improve analysis results.
- Expanding the dataset with additional healthcare-related variables.