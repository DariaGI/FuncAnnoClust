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
COPY . .
# Expose the port your Flask app runs on (adjust as necessary)
EXPOSE 5000

CMD ["conda", "run", "--no-capture-output", "-n", "funclust", "python", "FlaskApp/FlaskApp.py"]