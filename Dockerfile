FROM python:3.8-slim

WORKDIR .

COPY requirements.txt .
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8081

COPY . .

CMD ["python", "predict.py"]



