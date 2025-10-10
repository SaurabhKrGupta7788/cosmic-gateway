Cosmic Gateway: Interactive Exoplanet Classification
Cosmic Gateway is an interactive web platform designed to explore the fascinating intersection of astronomy and artificial intelligence. It provides a hands-on experience for users to classify exoplanets using a machine learning model and even tune the model's architecture in real-time.
 Core Features
Cosmic Gateway is built with an immersive, space-themed UI and offers a range of powerful features for both enthusiasts and learners.
1. Exoplanet Classification
KOI Data Submission: Users can input 13 distinct Kepler Object of Interest (KOI) parameters.
Instant Prediction: The backend, powered by a pre-trained neural network, provides an immediate classification of the exoplanet as either "Confirmed" or "Candidate."
2. Interactive Hyperparameter Tuning
Live Model Customization: A dedicated page allows users to visually and intuitively tune the core hyperparameters of a neural network, including:
Number of Hidden Layers
Neurons per Layer
Dropout Percentage
Learning Rate
Real-time Visualization: An animated neural network diagram updates in real-time to reflect the user's architectural changes.
Train Your Own Model: After tuning, users can train their custom model and use it for predictions within their session.
3. Immersive User Interface
Dynamic Space Theme: Features an animated Milky Way background with falling stars and parallax effects for a deeply immersive experience.
Glassmorphism Design: All UI components are designed with a sleek, semi-transparent "glass" effect, making the interface feel modern and futuristic.
🛠️ Technology Stack
The project is built using a modern stack for both the frontend and the backend.
Frontend:
HTML5
CSS3 (Flexbox, Grid, Custom Properties, Animations)
JavaScript (for dynamic visualizations on HTML Canvas)
Backend:
Python
Django Web Framework
TensorFlow / Keras (for building and training the neural networks)
Pandas & Scikit-learn (for data processing)
⚙️ Setup and Local Installation
To run this project on your local machine, follow these steps:
Clone the Repository
git clone [https://github.com/SaurabhKrGupta7788/cosmic-gateway.git](https://github.com/SaurabhKrGupta7788/cosmic-gateway.git)
cd cosmic-gateway


Create and Activate a Virtual Environment
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate


Install Dependencies
Make sure you have a requirements.txt file with all the necessary packages.
pip install -r requirements.txt


Run Django Migrations
python manage.py migrate


Start the Development Server
python manage.py runserver

The application will be available at http://1227.0.0.1:8000/.
✨ Contributors & Acknowledgements
This project was brought to life through a collaborative effort.
Project Development:
Coding Masters: Lead development and backend engineering.
📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

