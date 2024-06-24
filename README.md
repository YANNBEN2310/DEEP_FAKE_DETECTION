
# Reality Radar - Deepfake Detection

**Description:**

To combat fake news, the IA School entrusted us with this academic project to detect fake videos and images that proliferate on the internet and social networks. Our solution will produce a deep learning model capable of detecting them. Our product will be a mobile web application and a browser extension.

## Setting Up the Application

### Using Virtual Environment

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/YANNBEN2310/DEEP_FAKE_DETECTION.git
   cd <repository_directory>
   ```

2. **Create a Virtual Environment:**
   ```sh
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**
   - On Windows:
     ```sh
     venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```sh
     source venv/bin/activate
     ```

4. **Install the Required Packages:**
   ```sh
   pip install -r requirements.txt
   ```

5. **Run the Application:**
   ```sh
   python app.py
   ```

### Using Docker

1. **Clone the Repository:**
   ```sh
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Build the Docker Image:**
   ```sh
   docker build -t rr-deepfake-ym .
   ```

3. **Run the Docker Container:**
   ```sh
   docker run -p 5000:5000 rr-deepfake-ym
   ```

4. **Access the Application:**
   Open your web browser and navigate to `http://localhost:5000`

### Pretrained Model

- **Download Pretrained Model (.h5):** [Pretrained Model](https://drive.google.com/file/d/1SQrQZTjz419Ejp7qJiREV4g3UYKSk9T7/view)

### Iterations and Project Management

- **Iterations and Notebooks:** To view the different iterations and their corresponding notebooks, navigate to the `notebooks` folder in this repository.
- **Project Management Steps:** Detailed steps and progress updates for project management can be found in the `gestion_projet.md` file.

### Additional Information

- **Uploading Models:** When uploading a model, ensure the model file has a `.h5` extension.
- **Loading Media:** You can upload images or videos to scan for deepfake detection.

For further assistance, refer to the project documentation or contact the development team.
