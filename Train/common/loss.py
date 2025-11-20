import torch

def pairwise_logistic_loss(predictions, labels):
    """
    Calculates the pairwise logistic loss.

    Args:
        predictions (torch.Tensor): A tensor of shape (B, P) with model predictions.
        labels (torch.Tensor): A tensor of shape (B, P) with ground truth labels.

    Returns:
        torch.Tensor: A scalar tensor representing the mean loss.
    """
    # Get the batch size (B) and number of predictions per sample (P)
    B, P = predictions.shape

    # Expand dimensions to create all possible pairs
    # Shape becomes (B, P, 1) and (B, 1, P)
    predictions1 = predictions.unsqueeze(2)
    predictions2 = predictions.unsqueeze(1)

    labels1 = labels.unsqueeze(2)
    labels2 = labels.unsqueeze(1)

    # Calculate the difference between all pairs of predictions and labels
    # Shape for both will be (B, P, P)
    prediction_diff = predictions1 - predictions2
    label_diff = labels1 - labels2

    # Create a target tensor based on the sign of the label difference.
    # If label1 > label2, target is 1
    # If label1 < label2, target is -1
    # If label1 == label2, target is 0
    target = torch.sign(label_diff)

    # Calculate the logistic loss for each pair
    # We multiply prediction_diff by -target because we want to penalize
    # when the sign of prediction_diff does not match the sign of label_diff.
    # The softplus function is a smooth approximation of the relu function
    # and is equivalent to log(1 + exp(x)), which is the logistic loss.
    loss = torch.nn.functional.softplus(-target * prediction_diff)

    # To avoid self-comparison, we can zero out the diagonal of the loss matrix
    # This is an optional step but good practice.
    loss = loss.flatten(1)
    loss = loss[~torch.eye(P, dtype=torch.bool).flatten().unsqueeze(0).repeat(B, 1)]
    
    # Return the mean loss over all valid pairs
    return loss.mean()
