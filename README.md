\# AI Medical Imaging – Pneumonia Detection



\## Project Overview

An AI-powered medical imaging application that analyzes chest X-ray images and predicts NORMAL or PNEUMONIA using a Convolutional Neural Network (CNN).



\## Technologies Used

\- Python

\- TensorFlow and Keras

\- Streamlit

\- NumPy

\- Pillow

\- CNN



\## Features

\- Upload chest X-ray images

\- AI-based pneumonia prediction

\- Display NORMAL or PNEUMONIA results

\- Show model confidence score

\- Interactive web application



\## Model Evaluation

Preliminary evaluation using 20 test images:

\- Accuracy: 80%

\- Precision: 80%

\- Recall: 80%

\- F1-score: 80%



\## How to Run



Install dependencies:



```bash

pip install -r requirements.txt

```



Run the application:



```bash

python -m streamlit run app.py

```



\## Disclaimer

This is an educational project prototype. It is not a medical diagnostic tool and must not be used for clinical decisions.

\## Architecture Diagram



```mermaid

flowchart TD

&#x20;   A\["Chest X-ray Dataset"] --> B\["train\_model.py"]

&#x20;   B --> C\["CNN Training - TensorFlow/Keras"]

&#x20;   C --> D\["pneumonia\_model.keras"]

&#x20;   E\["User uploads X-ray"] --> F\["Streamlit app.py"]

&#x20;   D --> F

&#x20;   F --> G\["Resize image to 150x150"]

&#x20;   G --> H\["CNN Model Prediction"]

&#x20;   H --> I\["NORMAL"]

&#x20;   H --> J\["PNEUMONIA"]

&#x20;   I --> K\["Display result and model score"]

&#x20;   J --> K

```

