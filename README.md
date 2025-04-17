Name: Duen Michael Chombo
Roll: ID24m803
Assignment: 2

Wandbi:https://wandb.ai/duenchombo1-indian-institute-of-technology-madras/cnn/reports/-DA6401-INTRODUCTION-TO-DEEP-LEARNING--VmlldzoxMjI5MDk2OQ?accessToken=87s55tlt52xpcqyr0cnvuyffnaccpp0e3c63z4htndw9bvmt01znqawafynheuzj

Github:https://github.com/duenchombo/DA6401_ASSIGNMENT_2

Part A: Training from scratch
The first part of the code installs and imports necessary libraries for building and training deep learning models handling image data, tracking experiments (using a tool Weights & Biases), and visualizing results using matplotlib library
It then downloads a dataset Nature 12K, dataset which contains over 12,000 images of animals, plants, and fungi. The images are organized into training and validation folders, so the model can learn from some images and be tested on others

!curl -sSLO https://storage.googleapis.com/wandb_datasets/nature_12K.zip
!unzip -q nature_12K.zip

A list of the 10 biological classes (such as Insecta, Mammalia, and Reptilia) is created to help the model understand the categories it’s trying to learn.
classes=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']

The function prepare_dataset  prepares the image data for training a deep learning model. It begins by resizing all images to a standard size of 300x300 pixels and converting them into a format that the model can understand. To ensure proper training and evaluation, the dataset is split into two parts: 80% of the images are used for training the model, while the remaining 20% are reserved for validation to test the model's performance. In the end, the function provides two data pipelines—one for training and one for validation—that will be used to feed the data into the model during its learning process.

Part A Q1.
a small CNN model consisting of 5 convolution layers was build containing the parameters specified in question.


Part A Q2.
The model was trained on the dataset and the the sweep feature in wandb was used  find the best hyperparameter configuration
sweep_config = {
    'method': 'random',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'first_layer_filters': {'values': [32, 64]},
        'filter_org': {'values': [0.5, 1, 2]},
        'data_aug': {'values': [False, True]},
        'batch_norm': {'values': [True]},
        'dropout': {'values': [0.0, 0.2, 0.3]},
        'kernel_size': {'values': [3, 5]},
        'dense_size': {'values': [32, 64, 128]},
        'num_epochs': {'values': [10]},
        'num_classes': {'values': [num_classes]},  # Dynamic class count
    }
}

sweep_config['parameters']['num_classes'] = {'values': [num_classes]}

sweep_id = wandb.sweep(sweep_config, project='cnn')
wandb.agent(sweep_id, function=train_model)

Part A Q4
The best model was obtained using the following configuration which was looged in wandbi.
best_config = {
              'first_layer_filters': 64,
              'filter_org': 1,
              'data_aug': True,
              'batch_norm': True,
              'dropout': 0.3,
              'kernel_size': 5,
              'dense_size': 128,
              'activation': 'relu',
              'num_epochs': 10,
              'optimizer': 'adam',
              'conv_layers': 5,
              'num_classes':10,
          }
The test accuracy was Test Accuracy: 0.3455


Part B: Fine-tuning a pre-trained model

The flow I Followed:
1.	Import Required Packages
2.	Loading the Data
3.	Preparing Training and Validation Data (20% for test data)
4.	pre trained Model
5.	hyperparameter tuning using sweeps

i used the following sweep configuration
sweep_config = {'name': 'random-test-sweep', 'method': 'grid'}
sweep_config['metric'] = {'name': 'val_acc', 'goal': 'maximize'}
parameters_dict = {
                   'base_model_name': {'values': ["InceptionV3","ResNet50"]},
                   'tune': {'values': [False, True]},
                   'data_aug': {'values': [False, True]}
                  }
sweep_config['parameters'] = parameters_dict

you can run the sweep using the following code
sweep_id = wandb.sweep(sweep_config, project = 'cnn_part_B')
wandb.agent(sweep_id, function=pretrain_CNN_sweep)









