Project Proposal (Provisional)

This is the provisional project proposal for the smart traffic monitoring system using computer vision.

Question / Need:

What is the framing question of your analysis, or the purpose of the model/system you plan to build?

We aim to develop a smart traffic monitoring system that can automatically count the number of vehicles on the road and detect whether two-wheeler riders are wearing helmets or not. This system helps in monitoring traffic density and enforcing safety regulations by identifying helmet violations in real time.

Throughout this project, a deep learning-based object detection model such as YOLOv8 will be applied to detect vehicles and classify helmet usage based on video input.

The system will perform the following tasks:

Count total number of vehicles
Detect motorcycles and riders
Identify helmet and non-helmet cases
Who benefits from exploring this question or building this model/system?
Traffic police and law enforcement authorities
Smart city management systems
Road safety organizations
Data Description:

What dataset(s) do you plan to use, and how will you obtain the data?

The dataset used for this project is sourced from Kaggle, specifically a helmet detection dataset.

Dataset details:

Classes include:
Bike
Helmet
Without Helmet
Person
The dataset contains approximately 140 images

Due to the small dataset size, image augmentation techniques such as rotation, flipping, and scaling are applied to increase data diversity and reduce overfitting.

Additionally, pre-trained YOLO models are used to detect general vehicle classes such as cars, buses, and trucks.

If modeling, what will you predict as your target?

Number of vehicles (count)
Helmet detection (helmet / no helmet classification)
Tools:

How do you intend to meet the tools requirement of the project?

Ultralytics YOLO for training and detection
Streamlit for building the user interface
Hugging Face for model hosting and deployment
SQLite database for storing detection history
Numpy for numerical computations
Are you planning in advance to need or use additional tools beyond those required?
Pre-trained YOLO models for vehicle detection
Optional GPU support for faster inference
MVP Goal:

The expected outcome of the project is a functional smart traffic monitoring system capable of:

Counting vehicles from video input
Detecting helmet and non-helmet riders
Storing detection results in a database
Providing a simple user interface for monitoring

The final deliverables will include a trained model, a working application interface, and a project presentation
