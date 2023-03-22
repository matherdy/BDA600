import os
import random 
import numpy as np


from datasets import load_dataset



# Set the path to the folder
folder_path = "F:\Data\\small_train\\"

# Set the number of files you want to select
# num_files = 1000

# # Get a list of all files in the folder
# all_files = os.listdir(folder_path)

# # Randomly select a subset of the files
# subset_file_list = random.sample(all_files, num_files)

# # Get the full path for each file in the subset
# subset_file_list = [os.path.join(folder_path, file) for file in subset_file_list]

# 
#samp_train_data = load_dataset("image", data_files = subset_file_list)
samp_train_data = load_dataset("imagefolder", data_dir=folder_path)


from transformers import AutoFeatureExtractor, AutoModelForImageClassification, TrainingArguments, Trainer
#from evaluate import metric

extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

training_args = TrainingArguments(output_dir="test_trainer")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=samp_train_data,
    #eval_dataset=small_eval_dataset,
    #compute_metrics=compute_metrics,
)
print(samp_train_data[0])
#trainer.train()








#extra code

#from PIL import Image
# import torch
# from torchvision import transforms

# # Set the transformation to apply to the images
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # Load the images and transform them
# image_data = []
# for filename in os.listdir(subset_file_list):
#     if filename.endswith('.jpg'):
#         image_path = os.path.join(folder_path, filename)
#         image = Image.open(image_path)
#         image_tensor = transform(image)
#         image_data.append((image_tensor, filename))

# # Create a DataLoader to use for training
# batch_size = 32
# data_loader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=True)