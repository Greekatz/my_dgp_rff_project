import numpy as np
from . import loss

class TripletLoss(loss.Loss):
    def __init__(self, dout, margin=1.0):
        super().__init__(dout)
        self.margin = margin

    def eval(self, anchor, positive, negative):
        """
        Computes the triplet loss over a batch of embeddings.

        Parameters:
        - anchor: np.ndarray of shape (batch_size, embedding_dim)
        - positive: np.ndarray of shape (batch_size, embedding_dim)
        - negative: np.ndarray of shape (batch_size, embedding_dim)

        Returns:
        - loss: float, the average triplet loss over the batch
        """
        # Compute squared L2 distances
        pos_dist = np.sum((anchor - positive) ** 2, axis=1)
        neg_dist = np.sum((anchor - negative) ** 2, axis=1)

        # Compute triplet loss
        losses = np.maximum(pos_dist - neg_dist + self.margin, 0.0)

        # Return the average loss
        return np.mean(losses)

    def get_name(self):
        return "TripletLoss"