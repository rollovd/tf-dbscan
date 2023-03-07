# tf-dbscan
Pure Tensorflow 2+ DBSCAN algorithm

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

