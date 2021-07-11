# Sentiment Analysis
## By Rodrigo Guarneros

- To begin with, the user enters a review on our website.

- Next, our website sends that data off to an endpoint, created using API Gateway.

- Our endpoint acts as an interface to our Lambda function so our user data gets sent to the Lambda function.

- Our Lambda function processes the user data and sends it off to the deployed model's endpoint.

- The deployed model perform inference on the processed data and returns the inference results to the Lambda function.

- The Lambda function returns the results to the original caller using the endpoint constructed using API Gateway.

-  Lastly, the website receives the inference results and displays those results to the user.

Essentially, tuning a model means training a bunch of models, each with different hyperparameters, and then choosing the best performing model. Of course, we still need to describe two different aspects of hyperparameter tuning:

1) What is a bunch of models? In other words, how many different models should we train?

2) Which model is the best model? In other words, what sort of metric should we use in order to distinguish how well one model performs relative to another.

Generally speaking, the way to think about hyperparameter tuning inside of SageMaker is that we start with a base collection of hyperparameters which describe a default model. We then give some additional set of hyperparameters ranges. These ranges tell SageMaker which hyperparameters can be varied, with the goal being to improve the default model.

We then describe how to compare models, which in our instance is just by way of specifying a metric. Then we describe how many total models we want SageMaker to train.

Note: In addition to creating a tuned model in this notebook, we also saw how the attach method can be used to create an Estimator object which is attached to an already completed training job. This method is useful in other situations as well.

You will notice that throughout this module we train the same model multiple times. In most of the Boston Housing notebooks, for example, we train an XGBoost model with the same hyperparameters. The reason for this is so that each notebook is self contained and can be run even if you haven't run the other notebooks.

In your case however, you've probably already created an XGBoost model on the Boston Housing training set with the standard hyperparameters. If you wanted to, you could use the attach method to avoid having to re-train the model.

Now that you've seen an example of how to use SageMaker to tune a model, it's your turn to try it out!

Inside of the Mini-Projects folder is a notebook called IMDB Sentiment Analysis - XGBoost (Hyperparameter Tuning).ipynb. Inside of the notebook are some tasks for you to complete.

Note: To make things a little more interesting, there is a small error in the notebook. Try not to anticipate where the error is. Instead, just continue through the notebook as if nothing is wrong and when an error occurs, try to use CloudWatch to diagnose and fix it.

# Boston Housing In-Depth
Now we will look at creating a hyperparameter tuning job using the low level approach. Just like in the other low level approaches, this method requires us to describe the various properties that our hyperparameter tuning job should have.

To follow along, open up the Boston Housing - XGBoost (Hyperparameter Tuning) - Low Level.ipynb notebook in the Tutorials folder.

Creating a hyperparameter tuning job using the low level approach requires us to describe two different things.

The first, is a training job that will be used as the base job for the hyperparameter tuning task. This training job description is almost exactly the same as the standard training job description except that instead of specifying HyperParameters we specify StaticHyperParameters. That is, the hyperparameters that we do not want to change in the various iterations.

The second thing we need to describe is the tuning job itself. This description includes the different ranges of hyperparameters that we do want SageMaker to vary, in addition to the total number of jobs we want to have created and how to determine which model is best.

What have we learned so far?
In this lesson we took a look at how we can use SageMaker to tune a model. This can be helpful once we've found a good base model and we want to try and iterate a bit to refine our model and get a little more out of it.

We also looked at using CloudWatch to monitor our training jobs so that we can better diagnose errors when they arise. This can be especially helpful when training more complicated models such as those in which you can incorporate your own code.

What's next?
In the next lesson we will take a look at updating a deployed model. Once you've developed a model and deployed it the story is rarely over. What happens if some of the built in assumptions about your model change over time?

We will look at how you can create a new model which more accurately reflects the current state of your problem and then update an existing endpoint so that it uses your new model rather than the original one. In addition, using SageMaker, we can do this without needing to shut down the endpoint. This means that whatever application is using your deployed model won't notice any sort of disruption in service.

Depending on the application you have in mind for a particular machine learning model, accuracy may not always be the metric you wish to optimize. There may be some other constraints on getting the model to work in production. For example, your model may not be very easy to interpret or maybe performing inference for a particular model may be too costly.

In any case you may want to try alternative models. In the example we are working on here we construct a linear learner model as an alternative to the previously created XGBoost model.

Note: It is important to notice that the result returned by the linear learner model is json, compared to the csv data returned by the XGBoost model. You can't always assume that different models will return data in the same way although typically the return type is specified in the documentation.

Using the low level approach to creating endpoint configurations allows us to create endpoints that are more sophisticated. For example, endpoints which receive data and route that data to one of many different models. In the example here we are only using two different models but there may be situations in which you would want more.

In addition, SageMaker provides functionality to update an existing endpoint so that it conforms to a different endpoint configuration. Further, SageMaker does this in a way that does not require the existing endpoint to be shut down.

This is very beneficial as you may be working in an environment where there are other services that depend on your deployed endpoint.

Using the low level approach to creating endpoint configurations allows us to create endpoints that are more sophisticated. For example, endpoints which receive data and route that data to one of many different models. In the example here we are only using two different models but there may be situations in which you would want more.

In addition, SageMaker provides functionality to update an existing endpoint so that it conforms to a different endpoint configuration. Further, SageMaker does this in a way that does not require the existing endpoint to be shut down.

This is very beneficial as you may be working in an environment where there are other services that depend on your deployed endpoint.

In this mini-project we will take a look at situation in which we have a trained model which is working well, but then something changes with the underlying distribution on which our model is based. First we need to take a look at what might be the problem. Then we want to create a new, updated model and replace our old model without taking down the corresponding endpoint.

This mini-project notebook is called IMDB Sentiment Analysis - XGBoost (Updating a Model).ipynb and can be found inside of the Mini-Projects folder.

In this module we looked at various features offered by Amazon's SageMaker service. These features include the following.

Notebook Instances provide a convenient place to process and explore data in addition to making it very easy to interact with the rest of SageMaker's features.

Training Jobs allow us to create model artifacts by fitting various machine learning models to data.

Hyperparameter Tuning allow us to create multiple training jobs each with different hyperparameters in order to find the hyperparameters that work best for a given problem.

Models are essentially a combination of model artifacts formed during a training job and an associated docker container (code) that is used to perform inference.

Endpoint Configurations act as blueprints for endpoints. They describe what sort of resources should be used when an endpoint is constructed along with which models should be used and, if multiple models are to be used, how the incoming data should be split up among the various models.

Endpoints are the actual HTTP URLs that are created by SageMaker and which have properties specified by their associated endpoint configurations. Have you shut down your endpoints?

Batch Transform is the method by which you can perform inference on a whole bunch of data at once. In contrast, setting up an endpoint allows you to perform inference on small amounts of data by sending it do the endpoint bit by bit.

In addition to the features provided by SageMaker we used three other Amazon services.

In particular, we used S3 as a central repository in which to store our data. This included test / training / validation data as well as model artifacts that we created during training.

We also looked at how we could combine a deployed SageMaker endpoint with Lambda and API Gateway to create our own simple web app.

SageMaker Documentation
Developer Documentation can be found here: https://docs.aws.amazon.com/sagemaker/latest/dg/

Python SDK Documentation (also known as the high level approach) can be found here: https://sagemaker.readthedocs.io/en/latest/

Python SDK Code can be found on github here: https://github.com/aws/sagemaker-python-sdk

Setting up a Notebook Instance
The deployment project you will be working on is intended to be done using Amazon's SageMaker platform. In particular, it is assumed that you have a working notebook instance in which you can clone the deployment repository.

If you have not yet done this, please see the beginning of Lesson: Building a Model using SageMaker where we have walked you by creating a notebook and cloning the deployment repository. Alternatively, you can follow the instructions below.

Step 1. Go to AWS SageMaker
First, start by logging in to the AWS console, opening the SageMaker dashboard, and clicking on Create notebook instance.


AWS SageMaker → Notebook instances service

Step 2. Create a notebook instance
The Create notebook instance wizard will come up, asking you the following information:

Notebook instance settings - In this section, you may choose the notebook instance name of your choice. By default, a ml.t2.medium type is available. But, we will use ml.p2.xlarge for training a model and ml.m4.xlarge for deployment.
Note that your notebook may have a different name than the one displayed here.


Create notebook instance → Notebook instance settings

Permissions and encryption - Next, under IAM role field select Create a new role.

Create notebook instance → Permissions and encryption. Create a new IAM role

Create an IAM role - You should get a pop-up dialog box, where you have to select None radio-button under S3 buckets you specify field, as is shown in the image below.

Note that the IAM role name that appears may be different than the one displayed here.

Create an IAM role dialog box
Create an IAM role dialog box


Success, creating a new IAM role

Network - optional - Choose the No VPC option.

Create notebook instance → Network settings. Choose No VPC

Git repositories - Here you will clone the https://github.com/udacity/sagemaker-deployment.git repository to the current notebook instance only.

Create notebook instance → Git repositories setting

You're done! Click on Create notebook instance button.
Your notebook instance is now set up and ready to be used! Once the Notebook instance has loaded, you will see a screen similar to the following snapshot.


A successfully created notebook instance (Status: InService). You can access your notebook using the Open Jupyter Action.Setting up a Notebook Instance
The deployment project you will be working on is intended to be done using Amazon's SageMaker platform. In particular, it is assumed that you have a working notebook instance in which you can clone the deployment repository.

If you have not yet done this, please see the beginning of Lesson: Building a Model using SageMaker where we have walked you by creating a notebook and cloning the deployment repository. Alternatively, you can follow the instructions below.

Step 1. Go to AWS SageMaker
First, start by logging in to the AWS console, opening the SageMaker dashboard, and clicking on Create notebook instance.


AWS SageMaker → Notebook instances service

Step 2. Create a notebook instance
The Create notebook instance wizard will come up, asking you the following information:

Notebook instance settings - In this section, you may choose the notebook instance name of your choice. By default, a ml.t2.medium type is available. But, we will use ml.p2.xlarge for training a model and ml.m4.xlarge for deployment.
Note that your notebook may have a different name than the one displayed here.


Create notebook instance → Notebook instance settings

Permissions and encryption - Next, under IAM role field select Create a new role.

Create notebook instance → Permissions and encryption. Create a new IAM role

Create an IAM role - You should get a pop-up dialog box, where you have to select None radio-button under S3 buckets you specify field, as is shown in the image below.

Note that the IAM role name that appears may be different than the one displayed here.

Create an IAM role dialog box
Create an IAM role dialog box


Success, creating a new IAM role

Network - optional - Choose the No VPC option.

Create notebook instance → Network settings. Choose No VPC

Git repositories - Here you will clone the https://github.com/udacity/sagemaker-deployment.git repository to the current notebook instance only.

Create notebook instance → Git repositories setting

You're done! Click on Create notebook instance button.
Your notebook instance is now set up and ready to be used! Once the Notebook instance has loaded, you will see a screen similar to the following snapshot.


A successfully created notebook instance (Status: InService). You can access your notebook using the Open Jupyter Action.
Project Overview
Welcome to the SageMaker deployment project! In this project you will construct a recurrent neural network for the purpose of determining the sentiment of a movie review using the IMDB data set. You will create this model using Amazon's SageMaker service. In addition, you will deploy your model and construct a simple web app which will interact with the deployed model.

Project Instructions
The deployment project which you will be working on is intended to be done using Amazon's SageMaker platform. In particular, it is assumed that you have a working notebook instance in which you can clone the deployment repository.

Evaluation
Your project will be reviewed by a Udacity reviewer against the deployment project rubric. Review this rubric thoroughly, and self-evaluate your project before submission. All criteria found in the rubric must meet specifications for you to pass.

Project Submission
When you are ready to submit your project, collect all of the files in the project directory and compress them into a single archive for upload. In particular, make sure that the following files are included:

The SageMaker Project.ipynb file with fully functional code, all code cells executed and displaying output, and all questions answered.
An HTML or PDF export of the project notebook with the name report.html or report.pdf.
The completed train/train.py and serve/predict.py python files.
The edited website/index.html file.
Alternatively, your submission could consist of the GitHub link to your repository.

Project Submission Checklist
