import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

class PeerGroupAnalysisTool:
    def analyze_peer_group(self, input_data):
        """
        Perform peer group analysis using clustering.

        Args:
            input_data (list of dict or DataFrame): Input data containing peer group metrics.

        Returns:
            dict: A dictionary containing clusters and processed data.
        """
        # Convert input data to DataFrame if it's a list of dictionaries
        if isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        else:
            df = input_data

        # Standardize the data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

        # Impute missing values using KNN
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = imputer.fit_transform(df_scaled)

        # Perform clustering
        clustering = AgglomerativeClustering(n_clusters=3)
        cluster_labels = clustering.fit_predict(df_imputed)

        return {
            "clusters": cluster_labels.tolist(),
            "processed_data": df_imputed.tolist()
        }
