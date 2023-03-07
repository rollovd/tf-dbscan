import tensorflow as tf


class DBSCAN(tf.keras.Model):

    INPUT_TYPES = ['adjacency_matrix', 'feature_matrix']

    def __init__(self, eps: float = 0.4,
                 min_samples: int = 1,
                 input_type: str = 'adjacency_matrix'):
        """
        Initially this algorithm was implemented for specific Face Anti-Spoofing task.
        Therefore, some notations and terms will be devoted to spoofing terminology.

        Parameters
        ----------
        eps : float
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            This is not a maximum bound on the distances of points within a cluster.
            This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.

        min_samples : int
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            This includes the point itself.

        input_type : str List of input types:
        - `adjacency_matrix` - matrix NxN, where each cell is a distance (for instance, cosine similarity)
        between frames (where N is the number of frames)
        - `feature_matrix` - matrix NxM, where each frame is an embedding with the length M
        (and N is a number of frames)

        """
        super(DBSCAN, self).__init__()
        self._neighbours = None
        self._core_samples = None
        self._labels = None
        self._eps = eps
        self._min_samples = min_samples
        self._input_type = input_type
        assert self._input_type in self.INPUT_TYPES, "Incorrect string input type..."

    def _neighs(self, matrix):

        num_of_frames = tf.shape(matrix)[0]
        row_indices = tf.repeat(tf.range(1, num_of_frames + 1, dtype=tf.int32)[None],
                                tf.cast(num_of_frames, tf.int32), axis=0)

        relevant_indices = tf.cast(matrix > self._eps, dtype=tf.int32)
        mask_core_samples = tf.reduce_sum(relevant_indices, axis=1) >= self._min_samples

        indices = relevant_indices * row_indices - 1
        ragged_indices = tf.RaggedTensor.from_tensor(indices)
        mask = ragged_indices != -1

        return tf.ragged.boolean_mask(ragged_indices, mask), mask_core_samples

    def _update_queue(self, queue, sub_core):
        sub_core_neighbours = self._neighbours[sub_core]
        sub_core_labels = tf.gather(self._labels, sub_core_neighbours)
        sub_queue = tf.gather(sub_core_neighbours, tf.squeeze(tf.where(sub_core_labels == -1), axis=-1))
        queue = tf.concat([queue, sub_queue], axis=0)

        return queue

    @staticmethod
    def is_attack(num_of_clusters):
        return tf.cond(tf.less(num_of_clusters, 2),
                       lambda: tf.constant(False),
                       lambda: tf.constant(True))

    @staticmethod
    def _to_adjacency_matrix(feature_matrix):
        norm_feature_vectors = tf.transpose(tf.transpose(feature_matrix) / tf.norm(feature_matrix, axis=1))
        return norm_feature_vectors @ tf.transpose(norm_feature_vectors)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def call(self, input_matrix):

        adjacency_matrix = DBSCAN._to_adjacency_matrix(input_matrix) \
            if self._input_type == 'feature_matrix' \
            else input_matrix

        self._neighbours, self._core_samples = self._neighs(adjacency_matrix)
        num_of_frames = tf.shape(adjacency_matrix)[0]

        self._labels = -tf.ones(num_of_frames, dtype=tf.int32)
        indexes = tf.convert_to_tensor(tf.range(num_of_frames))
        current_label = tf.constant(0)

        filter_samples = tf.boolean_mask(indexes, self._core_samples)

        for core in filter_samples:

            queue = tf.convert_to_tensor([core])

            if self._labels[core] == -1:
                while tf.shape(queue)[0] > 0:
                    tf.autograph.experimental.set_loop_options(
                        shape_invariants=[(queue, tf.TensorShape([None]))],
                    )
                    sub_core = tf.gather(queue, tf.shape(queue)[0] - 1)
                    queue = tf.slice(queue, [0], [tf.shape(queue)[0] - 1])

                    self._labels = tf.tensor_scatter_nd_update(self._labels, [[sub_core]], [current_label])
                    queue = tf.cond(tf.equal(self._core_samples[sub_core], tf.constant(True)),
                                    lambda: self._update_queue(queue, sub_core),
                                    lambda: queue)

                current_label = tf.math.add(current_label, tf.constant(1))

        return tf.squeeze(self._labels)
        # return DBSCAN.is_attack(current_label)


if __name__ == "__main__":
    dbscan = DBSCAN(eps=0.4, min_samples=1)
    adjacency_matrix = tf.random.uniform((16, 16), dtype=tf.float32)
    labels = dbscan(adjacency_matrix)
