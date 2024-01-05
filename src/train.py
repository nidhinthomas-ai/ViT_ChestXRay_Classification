# train.py
import torch
import torch.optim as optim
import utils
import timm
import torchvision.utils as tvutils
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare data
train_dataset, val_dataset, _, _ = utils.prepare_data()

loader_train = DataLoader(train_dataset, batch_size=utils.batch_size, shuffle=True)
loader_val = DataLoader(val_dataset, batch_size=utils.batch_size, shuffle=False)

# Model
model = timm.create_model('vit_base_resnet50_224_in21k', pretrained=True, num_classes=utils.num_classes, img_size=utils.img_size)
model.to(device)

# Optimizer and loss
parameters = utils.add_weight_decay(model, 0.0001)
optimizer = optim.SGD(parameters, momentum=0.9, nesterov=True, lr=0.01)
loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)

# Learning rate scheduler
lr_scheduler = utils.StepLRScheduler(optimizer, decay_t=30, decay_rate=0.1, warmup_lr_init=0.0001, warmup_t=3)

num_epochs = 10
train_len = len(train_dataset)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(loader_train):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / train_len:.4f}')

    lr_scheduler.step(epoch + 1)

# Save the trained model
save_path = "path_to_destination/vit_10.pth"
torch.save(model.state_dict(), save_path)
