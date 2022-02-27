# What is this flower?
___

Web application for determining the type of flower 
bought in a flower shop from a photo.

## Technologies and Tools

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=for-the-badge&logo=Selenium&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Optuna](https://img.shields.io/badge/-OPTUNA-blue?style=for-the-badge&logo=appveyor?logo=appveyor)
![Matplotlib](https://img.shields.io/badge/-MATPLOTLIB-yellow?style=for-the-badge&logo=appveyor?logo=appveyor)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)

## Project Description

This project consists of three parts: creating a 
dataset, training a machine learning model, and 
creating a web application.<br>
The creation of a dataset consisting of photos of 
various types of flowers was carried out by a scraper 
implemented using **Selenium** and **Requests** 
libraries and collected in a package `scraping_dataset`. 
For more information about creating a dataset, 
see the [Data](#data) section.<br>
To classify flower images, a convolutional neural 
network of the **EfficientNet-B5** architecture 
was chosen as a machine learning model.
A package `model_utils` was created with auxiliary 
functions and classes implemented on the **Pytorch** 
framework to train the neural network.
The selection of model hyperparameters was performed 
using the **Optuna** library.
The learning process with visualizations is presented 
in `model_training_pipeline.ipynb`.<br>
For the accessible use of the trained model, a web 
application was created, written on the **Flask** 
microframework.
The web application is built in a **Docker** image 
and deployed to **AWS** via **CI/CD** pipeline using 
**Git Hub Actions**.<br>
You can see the working web application at the link 
[whatisthisflower.online](http://whatisthisflower.online)

## Project Structure

```bash
.
├── .github                                      # Service folder for GitHub Actions
│   └── workflows                                # Service folder for GitHub Actions
│       └── deploy_to_aws.yml                    # CI/CD pipeline for automatic deployment to AWS
├── predictive_model                             # Machine learning part of the project
│   ├── model_utils                              # Package for training neural networks
│   │   ├── __init__.py                          # Package initialization file
│   │   ├── hyperparameter_optimization.py       # Selection of neural network hyperparameters
│   │   ├── image_transformations.py             # Image transformations for input to a neural network
│   │   ├── images_visualization.py              # Visualizations associated with printing images
│   │   ├── load_dataset.py                      # Loading data into the format required for neural networks
│   │   ├── model_predictions.py                 # Neural network results
│   │   ├── model_quality.py                     # Checking the quality of a neural networks
│   │   └── model_training.py                    # Neural network training
│   ├── plots                                    # Images of the results of neural network hyperparameters
│   │   ├── plot_contour.png                     # selection for the report in model_training_pipeline.ipynb
│   │   ├── plot_contour_full.png
│   │   ├── plot_intermediate_values.png
│   │   ├── plot_intermediate_values_full.png
│   │   ├── plot_parallel_coordinate.png
│   │   ├── plot_parallel_coordinate_full.png
│   │   ├── plot_param_duration.png
│   │   ├── plot_param_duration_full.png
│   │   ├── plot_param_importances.png
│   │   ├── plot_param_importances_full.png
│   │   ├── tuning_result.png
│   │   └── tuning_result_full.png
│   ├── label_encoder.json                       # File with mappings predicted labels to flower names
│   ├── label_encoder.pkl                        # Binary file with mappings predicted labels to flower names
│   ├── model_training_pipeline.ipynb            # Process of training a neural network
│   └── model_weights.pth                        # Recorded weights of the trained neural network
├── scraping_dataset                             # Package for automatic collection of a dataset
│   ├── __init__.py                              # Package initialization file
│   ├── create_dataset.py                        # Downloads images corresponding to the list of subjects
│   ├── flower_types.json                        # List of required types of flowers
│   └── image_scraper.py                         # Collects links to images on a given topic
├── web_interface                                # Flask web application
│   ├── app                                      # Content of the web application
│   │   ├── static
│   │   │   ├── css
│   │   │   │   └── styles.css                   # Web application page styles
│   │   │   └── img
│   │   │       └── uploaded_image.png           # Example of an uploaded image
│   │   ├── templates
│   │   │   └── index.html                       # HTML template for the web application
│   │   ├── __init__.py                          # Initialization of the WSGI application
│   │   ├── forms.py                             # Forms used in the application
│   │   ├── model_inference.py                   # Performing flower classification on the transferred image
│   │   ├── utils.py                             # Business logic of the application
│   │   └── views.py                             # Routes used in the application
│   ├── run.py                                   # Launching the web application
│   └── web_app_requirements.txt                 # Dependencies are sufficient for a web application
├── .dockerignore                                # List of files that are not copied by Docker
├── .gitattributes                               # Parameters Git LFS (uploading large files)
├── .gitignore                                   # List of files that are not tracked by Git
├── aws-task-definition.json                     # Settings of EC2 instance on AWS
├── docker-compose.yml                           # Running a docker container with web applications
├── Dockerfile                                   # Creating a docker image of a web application
├── README.md                                    # Description of the project
└── requirements.txt                             # Dependencies for the entire project
```

## Data

An analysis of the available products of flower shops 
in the largest cities of Russia was carried out, 
as a result of which a list of flowers for sale 
was compiled recorded in `flower_types.json`.<br>
The total number of flower types was *63*.

The written package `scraping_dataset` contains a scraper
that, using the **Selenium** library, automatically 
downloads a specified number of images of each type from 
Google Image Search by going through the list 
in the JSON file.

To start the scraper, run the following command 
in the terminal:
```bash
python create_dataset.py -n 400
```
or, which is the same:
```bash
python create_dataset.py --n_images 400
```
The parameter `-n` (`--n_images`) takes as a value 
a positive integer and is responsible for the number 
of downloaded images of each type.<br>
In my case, the number of downloaded images of each type 
of flower was set to *400*.

After downloading all the images, manual cleaning of the 
dataset followed, which consisted in deleting 
inappropriate images or cropping the most suitable 
pieces of images.

The final dataset includes *14,532* images in total 
(*4.4* GB). All images are stored in JPG format, 
but have different resolutions. For each type of flower, 
there are from *76* to *327* images 
(the average number is *230*).<br>
The dataset was uploaded to the **Kaggle** platform 
([vitalymakin/flower-from-shops-classification](https://www.kaggle.com/vitalymakin/flower-from-shops-classification)), 
from where you can download it.

## How to Run the Web App

The business logic of determining the type of flower 
from a photo is wrapped in a web application written 
in the **Flask** microframework.

First, you need to create a file called `.env` in 
folder `web_interface/app`. The content of the file 
should be as follows:
```
FLASK_SECRET_KEY = 'come up with your own secret key'
```

To launch the web application, run the following 
commands in the terminal:
```bash
cd /web_interface
pip install -r web_app_requirements.txt
python run.py
```
or using Docker:
```bash
docker-compose up --build
```
Then open a web browser and go to http://localhost:5000.

## How to Use the Web App

The web application interface is shown in the 
screenshot below.

[![image](https://www.linkpicture.com/q/Снимок-экрана-2022-02-27-в-1.41.49.png)](https://www.linkpicture.com/view.php?img=LPic621ac9f996042888285153)

To upload an image, you need to click on the rectangular 
field with the inscription **"Please select some image"**, 
select a photo from your device and then click on the 
**"IDENTIFY FLOWER"** button.
After these actions, the uploaded image will be displayed, 
and the recognized type of flower will be signed below; 
in the first line, the name is in English, and the 
second - is in Russian.
Then you can repeat these steps.

You can see the working web application at the link 
[whatisthisflower.online](http://whatisthisflower.online)
