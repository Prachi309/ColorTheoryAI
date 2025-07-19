import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import gc
import psutil
import os

# Global model instance to avoid reloading
_model = None
_transform = None

def log_memory_usage(stage=""):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage {stage}: {memory_mb:.1f} MB")

def get_model():
    global _model, _transform
    if _model is None:
        log_memory_usage("before model load")
        
        # Load model only once
        model = models.resnet18(pretrained=True)
        num_classes = 4
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        # load saved state dictionary
        state_dict = torch.load('cp/best_model_resnet_ALL.pth', map_location=torch.device('cpu'))

        # create a new model with the correct architecture
        new_model = models.resnet18(pretrained=True)
        new_model.fc = nn.Linear(in_features, num_classes)
        new_model.load_state_dict(state_dict)

        # QUANTIZE MODEL TO REDUCE MEMORY USAGE
        print("🔄 Quantizing model to FP16 for memory optimization...")
        new_model = new_model.half()  # Convert to half precision (FP16)
        
        # Move to appropriate device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        new_model = new_model.to(device)
        
        # Set to evaluation mode
        new_model.eval()

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        _model = new_model
        _transform = transform
        
        log_memory_usage("after model load")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        log_memory_usage("after cleanup")
        
    return _model, _transform

def get_season(img):
    model, transform = get_model()
    
    log_memory_usage("before image processing")

    # Load and preprocess image
    image = Image.open(img).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    # Convert to half precision to match model
    device = next(model.parameters()).device
    image = image.half().to(device)

    model.eval()

    with torch.no_grad():
        output = model(image)
    
    pred_index = output.argmax().item()
    print("Decided color: ", pred_index)
    
    # Cleanup
    del image, output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    log_memory_usage("after image processing")
    
    return pred_index
