# tf-dbscan
Pure Tensorflow 2+ DBSCAN algorithm

Initially this DBSCAN implementation was developed for Face Recognition problem.
Therefore, this module assumes that a `feature_matrix` containing information about image embeddings will be fed into the algorithm or `adjacency_matrix`, which describes a **cosine distance** between embeddings.

Anyway, I hope this implementation will be useful and helpful a bit.

List of input types:
- `adjacency_matrix` - matrix NxN, where each cell is a distance (for instance, cosine similarity)
between frames (where N is the number of frames)
- `feature_matrix` - matrix NxM, where each frame is an embedding with the length M
(and N is a number of frames)

# Usage
```python
import tensorflow as tf
from dbscan import DBSCAN


dbscan = DBSCAN(eps=0.4, min_samples=1)
adjacency_matrix = tf.random.uniform((16, 16), dtype=tf.float32)
labels = dbscan(adjacency_matrix)
```

