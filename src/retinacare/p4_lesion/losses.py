"""Custom segmentation loss functions: Focal, Tversky, and Combined Focal-Tversky."""
import tensorflow as tf
from tensorflow.keras import backend as K


def tversky_loss(y_true, y_pred, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
    """Tversky loss: alpha penalizes false positives, beta penalizes false negatives (recall priority)."""
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)

    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1.0 - y_pred_pos))
    false_pos = K.sum((1.0 - y_true_pos) * y_pred_pos)

    tversky = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    return 1.0 - tversky


def focal_loss(y_true, y_pred, gamma: float = 2.0, alpha: float = 0.25):
    """Focal loss for addressing extreme foreground/background class imbalance."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = K.clip(y_pred_f, K.epsilon(), 1.0 - K.epsilon())

    pt_1 = tf.where(tf.equal(y_true_f, 1), y_pred_f, tf.ones_like(y_pred_f))
    pt_0 = tf.where(tf.equal(y_true_f, 0), y_pred_f, tf.zeros_like(y_pred_f))

    loss_1 = -K.sum(alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1))
    loss_0 = -K.sum((1.0 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0))
    return (loss_1 + loss_0) / (K.cast(K.shape(y_true_f)[0], tf.float32) + K.epsilon())


def combined_focal_tversky_loss(y_true, y_pred, class_weights=(1.5, 1.0, 1.0, 1.0)):
    """Combined 0.5*Focal + 0.5*Tversky loss across multi-label lesion channels."""
    # y_true, y_pred have shape (batch, H, W, 4) for [MA, HE, EX, SE]
    total_loss = 0.0
    num_classes = len(class_weights)

    for i in range(num_classes):
        yt_c = y_true[..., i]
        yp_c = y_pred[..., i]
        w_c = class_weights[i]

        l_focal = focal_loss(yt_c, yp_c, gamma=2.0)
        l_tversky = tversky_loss(yt_c, yp_c, alpha=0.3, beta=0.7)
        ch_loss = 0.5 * l_focal + 0.5 * l_tversky
        total_loss += w_c * ch_loss

    return total_loss / sum(class_weights)
