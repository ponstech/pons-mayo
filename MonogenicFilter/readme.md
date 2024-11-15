Model Architecture Details
Base Architecture: Multi-Scale CNN (with channels=3)
Custom Modification: A Monogenic Filter layer was added, which changes the input channels from 3 to 6. This adjustment is necessary to incorporate monogenic filter outputs into the model.
Pretrained Models
The trained model weights are provided as .pt files. These weights are optimized for the modified architecture (with channels=6).
Note: Ensure that the model architecture in your code matches this modified configuration when loading the provided .pt files.


Framework-Specific Libraries:
Python == 3.10.12
torch == 2.5.0+cu121
torchinfo == 1.8.0
torchvision == 0.20.0+cu121
pytorch_lightning == 2.4.0


Hardware Requirements
This project is optimized for GPUs. A Tesla T4 GPU or higher is recommended for optimal performance. 
