import torch

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x
    using PyTorch autograd.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    # Your code here - use autograd!
    # Hint: Create tensors with requires_grad=True, apply activation, call .backward()
    x =torch.tensor(x,dtype=torch.float32,requires_grad= True)

    der={}

    y= torch.sigmoid(x)
    y.backward()
    der["sigmoid"]=x.grad.item()

    x.grad.zero_()

    y=torch.tanh(x)
    y.backward()
    der["tanh"]=x.grad.item()

    x.grad.zero_()

    y=torch.relu(x)
    y.backward()
    der["relu"]=x.grad.item()

    return der