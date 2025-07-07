Game Clustering and Classification Project

Project Overview

This project implements a game clustering and classification model using CLIP for feature selection. It consists of:

Backend: Node.js (app.js)

Frontend: HTML, JavaScript, CSS

Functionality:

Clusters gameplay videos based on known labels

Assigns new gameplay videos to existing clusters

Uses KNN for classification

Displays results on a web-based frontend

How to Run the Project

1. Clone the Repository

git clone https://github.com/your-username/game-clustering.git
cd game-clustering

2. Install Dependencies

npm install

3. Start the Backend

node app.js

4. Start the Frontend

npm start

5. Access the Web App

Open your browser and go to:

http://localhost:3000

Project Structure

/game-clustering
├── backend
│   ├── app.js          # Node.js backend
│   ├── routes          # API routes
│   ├── models          # Data models
│   └── utils           # Helper functions
│
├── frontend
│   ├── index.html      # Main UI page
│   ├── GameplayClusteringUi.js # Frontend logic
│   ├── styles.css      # UI styling
│   └── assets          # Images, icons, etc.
│
├── data                # Dataset for clustering
├── README.txt          # Project documentation
└── package.json        # Node.js dependencies

Features

Extracts frames from gameplay videos (1 per second)

Assigns videos to clusters using KNN

Displays clustering visualization and stats

Shows progress bar while processing

Dependencies

Backend: Express, Node.js

Frontend: Vanilla JavaScript, HTML, CSS

Machine Learning: CLIP, KNN algorithm

Contribution

Feel free to contribute! Fork the repo and submit pull requests.

License

This project is licensed under the MIT License.
