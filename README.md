# Sentiment Analysis
## By Rodrigo Guarneros

- To begin with, the user enters a review on our website.

- Next, our website sends that data off to an endpoint, created using API Gateway.

- Our endpoint acts as an interface to our Lambda function so our user data gets sent to the Lambda function.

- Our Lambda function processes the user data and sends it off to the deployed model's endpoint.

- The deployed model perform inference on the processed data and returns the inference results to the Lambda function.

- The Lambda function returns the results to the original caller using the endpoint constructed using API Gateway.

-  Lastly, the website receives the inference results and displays those results to the user.
