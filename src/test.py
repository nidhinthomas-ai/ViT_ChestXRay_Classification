# test.py
import torch
import utils
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model_path = "path_to_destination/vit_10.pth"
model = timm.create_model('vit_base_resnet50_224_in21k', pretrained=False, num_classes=utils.num_classes, img_size=utils.img_size)
model.load_state_dict(torch.load(model_path))
model.to(device)

# Prepare test data
_, _, test_dataset, _ = utils.prepare_data()
loader_test = DataLoader(test_dataset, batch_size=utils.batch_size, shuffle=False)

# Test loop
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for images, labels in loader_test:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
