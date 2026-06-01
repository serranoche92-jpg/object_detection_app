<h1 align="center">Activity 3: Python Streamlit + ML Model</h1>
<h3 align="center">Real-Time Object Detection Using AI and Webcam</h3>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" />
  <img alt="YOLOv8" src="https://img.shields.io/badge/YOLOv8-Ultralytics-111111" />
  <img alt="Webcam" src="https://img.shields.io/badge/Webcam-Live%20Detection-0A66C2" />
  <img alt="Status" src="https://img.shields.io/badge/Project-Complete-2E7D32" />
</p>

<hr />

<h2>1. Project Summary</h2>

<p>
This activity is about building a <strong>Streamlit web application</strong> that detects real-time objects using Artificial Intelligence and a webcam.
The model used in this project is <code>yolov8n.pt</code>, which is the nano version of YOLOv8 designed for faster performance and lightweight devices.
</p>

<p>
The application captures live video from the webcam, processes each frame, and detects visible objects with labels and bounding boxes.
</p>

<hr />

<h2>2. Technologies Used</h2>

<ul>
  <li><strong>Python</strong> – Main programming language</li>
  <li><strong>Streamlit</strong> – Web application framework</li>
  <li><strong>OpenCV</strong> – Camera access and image processing</li>
  <li><strong>Ultralytics YOLOv8</strong> – Object detection model</li>
  <li><strong>NumPy</strong> – Array and image data handling</li>
</ul>

<hr />

<h2>3. Model Used</h2>

<p>
The model used is <code>yolov8n.pt</code>.
This is the smallest and fastest YOLOv8 model, making it suitable for laptops or lower-end devices.
</p>

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Speed</th>
      <th>Accuracy</th>
      <th>Best For</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>yolov8n.pt</code></td>
      <td>Fastest</td>
      <td>Basic</td>
      <td>Low-end devices / quick demos</td>
    </tr>
    <tr>
      <td><code>yolov8s.pt</code></td>
      <td>Fast</td>
      <td>Better</td>
      <td>Balanced performance</td>
    </tr>
    <tr>
      <td><code>yolov8m.pt</code></td>
      <td>Medium</td>
      <td>High</td>
      <td>Stronger laptops / GPUs</td>
    </tr>
  </tbody>
</table>

<hr />

<h2>4. Objects Tested</h2>

<p>
The following items were tested in front of the webcam:
</p>

<ul>
  <li>Bowl</li>
  <li>Cup</li>
  <li>Knife</li>
  <li>Scissors</li>
  <li>Cellphone</li>
</ul>

<hr />

<h2>5. Observations</h2>

<ul>
  <li>Some objects were detected correctly.</li>
  <li>Some items were not detected properly by the model.</li>
  <li>Small objects or unclear angles reduced detection accuracy.</li>
  <li>Lighting and camera quality may affect results.</li>
  <li>Some lag was noticed during real-time detection.</li>
</ul>

<hr />

<h2>6. Possible Reasons for Lag</h2>

<ul>
  <li>Laptop hardware performance limitations</li>
  <li>CPU usage during live webcam processing</li>
  <li>Multiple apps running in background</li>
  <li>Real-time AI inference workload</li>
</ul>

<hr />

<h2>7. Reflection</h2>

<p>
Based on my testing, <code>yolov8n.pt</code> works well for simple real-time detection,
but accuracy is lower compared to larger YOLO models.
It is fast and lightweight, but some objects may be missed or labeled incorrectly.
</p>

<p>
If better accuracy is needed, larger models like <code>yolov8s.pt</code> or <code>yolov8m.pt</code> can be used,
but they may require better hardware.
</p>

<hr />

<h2>8. How to Run</h2>

<pre><code>pip install streamlit ultralytics opencv-python numpy
streamlit run app.py
</code></pre>

<hr />

<h2>9. Conclusion</h2>

<p>
This activity helped demonstrate how AI can be used in object detection through a webcam using Streamlit.
It also showed the importance of choosing the right model depending on speed, accuracy, and device performance.
</p>
