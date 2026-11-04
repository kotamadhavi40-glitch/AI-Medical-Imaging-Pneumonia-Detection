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

&#x20;   A\[Chest X-ray Dataset] --> B\[Model Training]

&#x20;   B --> C\[CNN - TensorFlow/Keras]

&#x20;   C --> D\[Trained Model]

&#x20;   E\[User Uploads X-ray] --> F\[Streamlit Application]

&#x20;   D --> F

&#x20;   F --> G\[Image Preprocessing]

&#x20;   G --> H\[AI Prediction]

&#x20;   H --> I\[NORMAL]

&#x20;   H --> J\[PNEUMONIA]

&#x20;   I --> K\[Display Result]

&#x20;   J --> K

&#x20;   L\[Model Evaluation] --> M\[Accuracy, Precision, Recall, F1]
## Application Screenshots

### Home Page
![Application Home Page](screenshots/home-page.png)

### Pneumonia Prediction
![Pneumonia Prediction](screenshots/pneumonia-result.png)

### Normal Prediction
![Normal Prediction](screenshots/normal-result.png)

