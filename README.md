# Data_Science_Workshop

A deep learning-powered jewelry recommendation system that uses CLIP embeddings and advanced clustering techniques to provide personalized jewelry recommendations based on user preferences.

## 🌟 Features

- **CLIP-based Image Embeddings**: Uses OpenAI's CLIP model to generate rich semantic representations of jewelry images
- **Multiple Clustering Approaches**: Supports both traditional K-Means and advanced DEPICT (Deep Embedded Clustering) methods
- **Dimensionality Reduction**: Implements UMAP for better clustering in lower-dimensional spaces
- **Interactive Rating System**: Allows users to rate jewelry items to build personalized profiles
- **Personalized Recommendations**: Generates recommendations based on user preferences and cluster affinities
- **Model Persistence**: Save and load trained models to/from Google Drive

## 🏗️ Architecture

The system consists of several key components:

1. **Data Pipeline**: Downloads and processes jewelry images from Hugging Face datasets
2. **Feature Extraction**: Uses CLIP to generate 512-dimensional embeddings for each jewelry item
3. **Clustering Module**: Groups similar items using either:
   - K-Means clustering on raw embeddings
   - UMAP + K-Means for dimensionality reduction
   - DEPICT (Variational Autoencoder + Clustering) for deep clustering
4. **Recommendation Engine**: Generates personalized recommendations based on user rating history
5. **Interactive Interface**: Provides functions for rating items and displaying recommendations

## 📋 Requirements

```python
# Core dependencies
torch
torchvision
clip-by-openai
pandas
numpy
matplotlib
scikit-learn
umap-learn
tqdm
requests
Pillow

# For interactive features
jupyter
ipywidgets

# For data handling
pyarrow  # for parquet files
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install core dependencies
!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git
!pip install umap-learn scikit-learn pandas matplotlib numpy Pillow requests
```

### 2. Load the Dataset

The system automatically downloads the jewelry dataset from Hugging Face:

```python
# Dataset URLs are built into the notebook
parquet_urls = [
    "https://huggingface.co/api/datasets/ayesha111/dataset_jewellery1/parquet/default/train/0.parquet",
    # ... additional URLs
]
```

### 3. Basic Usage

```python
# Load embeddings (or compute them if first time)
embeddings_df = pd.read_pickle("/path/to/embeddings.pkl")
original_df = load_parquet_dataset(parquet_urls)

# Create and train a simple K-Means model
kmeans_model = KMeansModelWrapper(n_clusters=11)
kmeans_model.fit(embeddings_df.values)

# Create recommender
recommender = JewelryRecommender(kmeans_model, embeddings_df, original_df)

# Rate some items interactively
rated_items = interactive_jewelry_rating(recommender, original_df)

# Get personalized recommendations
recommendations = display_my_recommendations(recommender, original_df)
```

## 🔧 Detailed Usage

### Computing CLIP Embeddings

If running for the first time, the system will compute CLIP embeddings for all jewelry images:

```python
# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Embeddings are computed in batches and saved
batch_size = 128
# ... (automatic batch processing code)
```

### Choosing a Clustering Method

#### Option 1: K-Means (Fastest)
```python
kmeans_model = KMeansModelWrapper(n_clusters=11)
kmeans_model.fit(embeddings)
```

#### Option 2: UMAP + K-Means (Better clusters)
```python
# Reduce dimensions with UMAP
reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.1)
umap_embeddings = reducer.fit_transform(embeddings)

# Then apply K-Means
kmeans_model = KMeansModelWrapper(n_clusters=11)
kmeans_model.fit(umap_embeddings)
```

#### Option 3: DEPICT (Most sophisticated)
```python
# Train deep clustering model
model, recommender, losses = main(
    embeddings_df=umap_df,
    original_df=original_df,
    n_clusters=8,
    hidden_dim=32,
    latent_dim=16,
    epochs=50
)
```

### Interactive Rating

The system provides an interactive rating interface:

```python
# Rate jewelry items (1-5 stars)
rated_items = interactive_jewelry_rating(recommender, original_df, my_user_id="your_id")
```

Users can rate items on a 1-5 scale, and the system builds preference profiles based on cluster affinities.

### Getting Recommendations

```python
# Get top 5 recommendations
rec_indices, scores = recommender.recommend_for_user("your_id", top_n=5)

# Display recommendations with images
display_my_recommendations(recommender, original_df, "your_id")
```

## 📊 Model Performance

### Clustering Quality Metrics

The notebook includes comprehensive evaluation of clustering quality:

- **Silhouette Score**: Measures cluster separation and cohesion
- **Elbow Method**: Finds optimal number of clusters
- **Visual Inspection**: UMAP projections for cluster visualization

### Typical Results

- **Raw CLIP + K-Means**: Silhouette score ~0.11
- **UMAP + K-Means**: Silhouette score ~0.53
- **DEPICT**: Learns optimal representations with reconstruction + clustering loss

## 💾 Model Persistence

### Saving Models

```python
# Save to Google Drive
save_model_to_drive(model, recommender, losses, "/content/drive/MyDrive/jewelry_recommender")
```

### Loading Models

```python
# Load from Google Drive
model, recommender = load_model_from_drive("/content/drive/MyDrive/jewelry_recommender")
```

## 📁 File Structure

```
jewelry_recommender/
├── depict_model.pt              # PyTorch model weights
├── model_config.pkl             # Model architecture config
├── recommender.pkl              # Recommender object
├── training_losses.pkl          # Training loss history
├── cluster_assignments.npy      # Item cluster assignments
├── latent_vectors.npy          # Latent representations
├── cluster_probs.npy           # Soft cluster probabilities
├── embeddings_with_clusters.pkl # Embeddings + cluster info
└── README.md                   # Usage instructions
```

## 🎯 Key Classes and Functions

### Core Classes

- **`DEPICT`**: Variational autoencoder with integrated clustering
- **`JewelryRecommender`**: Main recommendation engine
- **`KMeansModelWrapper`**: Wrapper for sklearn K-Means compatibility

### Key Functions

- **`train_depict()`**: Train the DEPICT clustering model
- **`interactive_jewelry_rating()`**: Interactive rating interface
- **`display_my_recommendations()`**: Show personalized recommendations
- **`main()`**: End-to-end training pipeline

## 🔬 Technical Details

### DEPICT Model Architecture

The DEPICT model combines:
- **Variational Autoencoder**: For learning compact representations
- **Clustering Layer**: Student's t-distribution for soft assignments
- **Joint Training**: Simultaneous reconstruction and clustering optimization

### Recommendation Algorithm

1. **User Profiling**: Track average ratings per cluster
2. **Cluster Scoring**: Weight clusters by user preferences
3. **Item Ranking**: Score items by cluster affinity and centroid distance
4. **Diversity**: Ensure recommendations span top preferred clusters

### Optimization Features

- **Adaptive Architecture**: Automatically adjusts network size based on input dimensions
- **Gradient Clipping**: Prevents training instability
- **Early Stopping**: Avoids overfitting with patience-based stopping
- **Learning Rate Scheduling**: Reduces learning rate when loss plateaus

## 🎨 Visualization

The system includes rich visualization capabilities:
- **Cluster Distribution**: Shows item counts per cluster
- **Rating Heatmaps**: Visualizes user preferences by cluster
- **Recommendation Display**: Shows recommended items with images and descriptions
- **Training Curves**: Plots loss evolution during training

## 🔍 Usage Tips

1. **First Time Setup**: Allow time for CLIP embedding computation (can take 30+ minutes for large datasets)
2. **Cluster Selection**: Use silhouette analysis to choose optimal cluster count
3. **Rating Strategy**: Rate diverse items across clusters for better recommendations
4. **Model Choice**: Use UMAP + K-Means for best balance of speed and quality
5. **Memory Management**: Use batch processing for large datasets to avoid OOM errors

## 🤝 Contributing

The system is designed to be extensible:
- Add new clustering algorithms by implementing the model interface
- Extend recommendation algorithms in the `JewelryRecommender` class
- Add new evaluation metrics in the analysis sections

## 📝 License

This project is designed for educational and research purposes. Please respect the original jewelry dataset license and CLIP model terms of use.

## 🙏 Acknowledgments

- OpenAI for the CLIP model
- Hugging Face for hosting the jewelry dataset
- The scikit-learn and PyTorch communities for excellent ML libraries
