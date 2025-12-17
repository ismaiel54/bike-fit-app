# 🚴 Bike Fit Analyzer

An AI-powered bike fit tool that analyzes cycling posture from video using computer vision and pose estimation. Get personalized bike fit recommendations based on your riding position, bike type, and fitness goals.

## ✨ Features

- **Video Analysis**: Upload cycling videos (MP4, MOV) to analyze your riding posture
- **Pose Detection**: Uses MediaPipe to detect and track key body landmarks
- **Angle Measurements**: Calculates critical bike fit angles:
  - Knee angle
  - Hip angle
  - Foot angle
  - Torso angle
  - Elbow angle
- **Bike Type Support**: Optimized recommendations for:
  - Road bikes
  - Gravel/CX bikes
  - Mountain bikes (MTB)
  - Triathlon/Time Trial bikes
  - Spin/Exercise bikes
- **Personalized Recommendations**: 
  - Customizable fit goals (Comfort, Balanced, Aero-Performance)
  - Mobility assessment integration
  - Priority-based adjustment suggestions
- **Visual Output**: Generates annotated videos and images with angle overlays
- **Stroke Analysis**: Analyzes pedal stroke positions (top, mid, bottom)

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **Computer Vision**: MediaPipe, OpenCV
- **Frontend**: HTML5, CSS3, JavaScript
- **Dependencies**: NumPy, Jinja2

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bike-fit-app
   ```

2. **Navigate to the backend directory**
   ```bash
   cd backend
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

1. **Start the server**
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **Open your browser**
   Navigate to `http://127.0.0.1:8000`

3. **Upload a video**
   - Select your bike type and riding goal
   - Optionally fill in mobility assessment and notes
   - Upload a video file (MP4 or MOV format)
   - Click "Analyze Video"

4. **Review results**
   - View annotated video with pose overlays
   - Check angle measurements and fit windows
   - Review personalized recommendations
   - See prioritized adjustment suggestions

## 📹 Video Requirements

- **Format**: MP4 or MOV
- **Orientation**: Landscape preferred (portrait videos are auto-rotated)
- **Content**: Rider should be visible from the left side
- **Quality**: Clear view of the rider's full body

## 📊 Understanding the Results

### Fit Windows
Each angle is compared against target ranges for your bike type and goal:
- **In Range**: ✅ Optimal position
- **Slightly Off**: ⚠️ Minor adjustments recommended
- **Off**: ❌ Significant adjustment needed

### Recommended Actions
Prioritized list of adjustments to try:
1. **Priority 1**: Critical adjustments (e.g., saddle height)
2. **Priority 2**: Important refinements (e.g., bar position)
3. **Priority 3**: Fine-tuning adjustments

### Stroke Samples
Estimated angles at different pedal stroke positions:
- **Bottom of Stroke**: Leg extended (max knee angle)
- **Mid Stroke**: Transition position
- **Top of Stroke**: Leg flexed (min knee angle)

## ⚙️ Configuration

### Bike Types
Each bike type has optimized angle ranges:
- **Road**: Balanced fit for power, comfort, and endurance
- **Gravel**: Slightly more upright for stability and mixed terrain
- **MTB**: Upright fit for technical terrain and control
- **TT/Triathlon**: Aggressive aero position prioritizing low drag

### Fit Goals
- **Comfort**: Wider acceptable ranges, prioritizes rider comfort
- **Balanced**: Standard ranges for general riding
- **Aero-Performance**: Narrower ranges, optimized for aerodynamics

## 📁 Project Structure

```
bike-fit-app/
├── backend/
│   ├── main.py              # FastAPI application and routes
│   ├── pose_analysis.py     # Pose detection and analysis logic
│   ├── requirements.txt     # Python dependencies
│   ├── templates/
│   │   └── index.html       # Frontend UI
│   ├── static/              # Static files
│   └── outputs/             # Generated videos and images
├── README.md
└── LICENSE
```

## 🔧 API Endpoints

### `GET /`
Serves the main bike-fit UI.

### `POST /analyze-video`
Analyzes uploaded video and returns bike fit recommendations.

**Request**:
- `file`: Video file (MP4/MOV)
- `bike_type`: Bike type selection
- `goal`: Fit goal (Comfort/Balanced/Aero-Performance)
- `mobility`: JSON string with mobility scores
- `notes`: Optional notes/pain points

**Response**:
- Pose detection results
- Angle measurements
- Fit windows
- Recommendations
- Annotated video/image URLs

## ⚠️ Disclaimer

This tool provides video-based estimates for bike fit analysis. It's designed for iterative adjustments and general guidance. For pain, injury, or professional bike fitting, please consult a certified bike fitter or healthcare professional.

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues, questions, or suggestions, please open an issue on the repository.

---

**Note**: This tool analyzes posture from video and provides recommendations based on biomechanical principles. Individual results may vary based on flexibility, injury history, and personal preferences.
