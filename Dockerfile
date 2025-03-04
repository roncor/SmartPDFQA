# Use a smaller base image for efficiency
FROM python:3.10-slim AS builder

# Set static user and group for the non-root user
ARG USER_NAME=llm_usr
ARG GROUP_NAME=llm_usr

# Install essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -r $GROUP_NAME && useradd -r -g $GROUP_NAME -m -s /bin/bash $USER_NAME

# Set the working directory
WORKDIR /home/$USER_NAME/app

# Copy requirements file
COPY ./requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and only download Chromium
RUN pip install playwright && playwright install chromium --with-deps

# Copy application files
COPY . .

# Change ownership to the non-root user
RUN chown -R $USER_NAME:$GROUP_NAME /home/$USER_NAME/app

# Expose port 5001
EXPOSE 5001

# Switch to the non-root user
USER $USER_NAME

# Start an interactive bash shell
CMD ["bash"]
