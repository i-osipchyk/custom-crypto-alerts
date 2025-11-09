# Use AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Install system build tools for packages like numpy
RUN yum update -y && \
    yum groupinstall "Development Tools" -y && \
    yum install -y gcc gcc-c++ gcc-gfortran blas-devel lapack-devel && \
    yum clean all

# Upgrade pip
RUN pip install --upgrade pip

# Copy bot code and requirements
COPY main.py functions.py requirements.txt ./

# Install Python dependencies
RUN pip install -r requirements.txt

# Set the Lambda handler
CMD ["main.lambda_handler"]
