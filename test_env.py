import torch

x = torch.rand(3)
print(x)

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Print the name of each GPU
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Set the device to the first GPU (you can change it based on your requirements)
    device = torch.device("cuda:0")

    # Run a simple test on the GPU
    x = torch.rand(3, 3).to(device)
    y = torch.rand(3, 3).to(device)
    z = x + y

    print("Test result on GPU:")
    print(z)

else:
    print("CUDA is not available. Running on CPU.")
    device = torch.device("cpu")

