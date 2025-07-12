import torch as pt

def current_device():
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def to_device(tensor, device):
    return tuple(t.to(device) for t in tensor)

