# Early Detection of Breast Cancer Using Machine Learning

## Overview
This project uses the latest advancements in **medical diagnostics** and **machine learning** to enable early detection of breast cancer. Early detection significantly improves the chances of successful treatment and patient survival.

## Technologies Used
- **Scikit-learn**: For building ML models (Random Forest Classifier)
- **Pandas & NumPy**: Data handling and analysis
- **Seaborn & Matplotlib**: Visualization of tumor classifications
- **Joblib**: Model serialization
- **StandardScaler**: Feature scaling for better model performance

## Dataset
The project uses the **Breast Cancer Wisconsin Diagnostic Dataset**, which includes modern diagnostic features such as:
- Radius, Texture, Perimeter, Area
- Smoothness, Compactness, Symmetry, Fractal dimension

## AI + Healthcare
This model mimics the type of analysis done by AI-based tools in modern hospitals. When integrated with imaging data or patient health records, it can act as a **clinical decision support system (CDSS)** for doctors.

## Future Scope
- Integration with **Deep Learning** (e.g., mammogram image recognition using CNNs)
- API deployment for hospitals and mobile health apps
- Real-time risk assessment dashboards for medical professionals

## Running the Project
```bash
python breast_cancer_detection.py
```

This will train the model and output diagnostic performance with a saved `.pkl` model file.

## License
This project is open-source and available for educational and research purposes.
