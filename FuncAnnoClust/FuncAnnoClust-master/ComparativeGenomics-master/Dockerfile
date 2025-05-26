FROM conda/miniconda3:latest
# Update conda (if needed)
RUN conda update -n base -c defaults conda
# Set the working directory
WORKDIR /app
# Copy the environment file
COPY environment.yml .
# Create the Conda environment
RUN conda env create -f environment.yml
# Activate the environment
RUN echo "conda activate funclust" >> ~/.bashrc
