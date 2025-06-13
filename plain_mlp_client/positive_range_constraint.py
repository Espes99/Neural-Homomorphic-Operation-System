import tensorflow as tf
class PositiveRangeConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        # First ensure weights are positive, then clip to max range
        return tf.clip_by_value(tf.maximum(0., w), self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}
