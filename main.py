from kivy.lang import Builder
from kivy.core.window import Window
from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.menu import MDDropdownMenu
from kivy.clock import Clock
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import requests
import numpy as np
from sklearn.cluster import KMeans

# Set fixed mobile layout size
Window.size = (412, 917)

KV = """
MDScreenManager:
    WelcomeScreen:
    UploadScreen:
    ResultsScreen:
    WeatherScreen:

<WelcomeScreen>:
    name: "welcome"
    MDFloatLayout:
        Image:
            source: "background.jpg"
            size: self.size
            allow_stretch: True
            keep_ratio: False
            opacity: 0.5

        MDLabel:
            text: "Welcome to\\n[font=Roboto-Bold]Soil Scan[/font]"
            markup: True
            halign: "center"
            font_style: "H4"
            pos_hint: {"center_y": 0.6}

        MDRaisedButton:
            text: "Continue"
            pos_hint: {"center_x": 0.5, "center_y": 0.4}
            size_hint: None, None
            size: 180, 50
            on_release: root.manager.current = "upload"

<UploadScreen>:
    name: "upload"
    MDFloatLayout:
        Image:
            source: "background.jpg"
            size: self.size
            allow_stretch: True
            keep_ratio: False
            opacity: 0.5

        MDIconButton:
            icon: "menu"
            id:open_menu_button
            pos_hint: {"top": 1, "left": 0.2}
            size_hint: None, None
            size: "55dp", "55dp"
            on_release: root.open_menu()

        MDCard:
            id: upload_card
            orientation: "vertical"
            size_hint: None, None
            size: "200dp", "200dp"
            pos_hint: {"center_x": 0.5, "center_y": 0.7}
            elevation: 10
            padding: "10dp"
            spacing: "10dp"
            opacity: 0

            Image:
                id: uploaded_image
                source: "default.png"
                size_hint: None, None
                size: 150, 150
                pos_hint: {"center_x": 0.5, "center_y": 1}

        MDRaisedButton:
            text: "Upload Soil Image"
            pos_hint: {"center_x": 0.5, "center_y": 0.5}
            size_hint: None, None
            size: 200, 50
            on_release: root.open_file_manager()

        MDRaisedButton:
            text: "Scan"
            pos_hint: {"center_x": 0.5, "center_y": 0.4}
            size_hint: None, None
            size: 180, 50
            on_release: root.analyze_image()

<ResultsScreen>:
    name: "results"
    MDFloatLayout:
        Image:
            source: "background.jpg"
            size: self.size
            allow_stretch: True
            keep_ratio: False
            opacity: 0.5

        MDIconButton:
            icon: "arrow-left"
            pos_hint: {"top": 1, "left": 0}
            size_hint: None, None
            size: "55dp", "55dp"
            on_release: app.go_back_to_upload()

        MDLabel:
            id: result_label
            text: ""
            theme_text_color: "Custom"
            text_color: 0, 0, 0, 1
            font_style: "H4"
            halign: "center"
            pos_hint: {"center_x": 0.5, "center_y": 0.7}
            size_hint: None, None
            width: 300

        MDLabel:
            id: ph_label
            text: ""
            halign: "center"
            font_style: "H5"
            pos_hint: {"center_y": 0.6}

        MDLabel:
            id: crop_label
            text: ""
            halign: "center"
            font_style: "H5"
            pos_hint: {"center_y": 0.6}

        MDLabel:
            id: crop_label
            text: "Recommended Crops: "
            font_style:"H5"
            color: 0, 0, 0, 1  # Green color
            pos_hint: {"center_y": 0.5}


<WeatherScreen>:
    name:"weather"
    MDFloatLayout:

        Image:
            source: "background.jpg"
            size: self.size
            allow_stretch: True
            keep_ratio: False
            opacity: 0.5

        MDIconButton:
            icon: "menu"
            id:open_menu_button
            pos_hint: {"top": 1, "left": 0.2}
            size_hint: None, None
            size: "55dp", "55dp"
            on_release: root.open_menu()

        MDLabel:
            text: "Weather Information"
            font_size: "20sp"
            size_hint_y: None
            height: "40dp"
            halign: "center"
            pos_hint: {"center_x": 0.5,"center_y": 0.8}
        
        MDTextField:
            id: city_input
            hint_text: "Enter City"
            size_hint_y: None
            size_hint_x: None
            height: "20dp"
            width: "100dp"
            pos_hint: {"center_x": 0.5,"center_y": 0.6}
        
        MDRaisedButton:
            text: "Get Weather"
            size_hint: None, None
            size: 200, 50
            pos_hint: {"center_x": 0.5,"center_y": 0.5}
            on_release: root.fetch_weather()

        MDLabel:
            id: weather_label
            text: ""
            size_hint_y: None
            height: "80dp"
            halign: "center"
            pos_hint: {"center_x": 0.5,"center_y": 0.4}
        
"""

# -------------------------------
# Define Model for Soil Classification & pH Prediction
# -------------------------------
class SoilClassificationWithPH(nn.Module):
    def __init__(self, base_model, num_classes=4):
        super(SoilClassificationWithPH, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC layer
        self.fc_class = nn.Linear(base_model.fc.in_features, num_classes)  # Soil type classification
        self.fc_regress = nn.Linear(base_model.fc.in_features, 1)  # pH level regression

    def forward(self, x):
        features = self.base_model(x)  # Extract features
        features = torch.flatten(features, 1)  # Flatten output
        
        class_output = self.fc_class(features)  # Soil type prediction
        ph_output = self.fc_regress(features)  # pH level prediction
        
        return class_output, ph_output, features

# -------------------------------
# Load Pretrained Model (Only Once)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = models.resnet18(pretrained=False)
model = SoilClassificationWithPH(base_model)
model.load_state_dict(torch.load("soil_ph_classification_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Define Soil Classes & pH Labels
soil_classes = ["Alluvial", "Black", "Clay", "Red"]
ph_labels = {
    0: "Acidic",
    1: "Neutral",
    2: "Alkaline"
}

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Train K-Means on Training Data Features (Only Once)
# -------------------------------
def train_kmeans(data_path, num_clusters=3):
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    feature_list = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            _, _, features = model(images)
            feature_list.append(features.cpu().numpy())

    feature_matrix = np.vstack(feature_list)  # Convert to NumPy array
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(feature_matrix)  # Train K-Means
    return kmeans

# Train K-Means (only needs to be done once)
kmeans = train_kmeans("./SoilDatasets/train_data")

# Crop Recommendation Dictionary Based on Soil Type & pH
crop_recommendations = {
    "Alluvial": {
        "Acidic": ["Tea", "Rubber", "Pineapple"],
        "Neutral": ["Rice", "Wheat", "Sugarcane"],
        "Alkaline": ["Barley", "Oats", "Peas"]
    },
    "Black": {
        "Acidic": ["Cotton", "Soybean", "Tobacco"],
        "Neutral": ["Maize", "Groundnut", "Sunflower"],
        "Alkaline": ["Sorghum", "Millets", "Sesame"]
    },
    "Clay": {
        "Acidic": ["Coffee", "Cocoa", "Spices"],
        "Neutral": ["Paddy", "Sugarcane", "Jute"],
        "Alkaline": ["Lentils", "Peas", "Mustard"]
    },
    "Red": {
        "Acidic": ["Tapioca", "Pineapple", "Citrus"],
        "Neutral": ["Tomato", "Chili", "Onion"],
        "Alkaline": ["Groundnut", "Ragi", "Castor"]
    }
}
# -------------------------------
# Function to Predict Soil Type & pH
# -------------------------------


# Function to Predict Soil Type, pH Level, and Recommended Crops
def predict_soil_and_ph(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        class_output, ph_output, features = model(image)

        # Get Soil Type
        class_pred = torch.argmax(class_output, dim=1).item()
        soil_type = soil_classes[class_pred]

        # Get pH Value & Category
        ph_value = round(ph_output.item(), 2)
        feature_vector = features.cpu().numpy().reshape(1, -1)
        cluster_label = kmeans.predict(feature_vector)[0]
        ph_category = ph_labels[cluster_label]  # Acidic, Neutral, Alkaline

        # Recommend Crops Based on Soil & pH
        recommended_crops = crop_recommendations[soil_type][ph_category]

    return soil_type, ph_value, ph_category, recommended_crops




# -------------------------------
# Kivy Screen Definitions
# -------------------------------
class WelcomeScreen(Screen):
    pass

class UploadScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_manager = MDFileManager(select_path=self.select_file)
        self.selected_image_path = ""

    def open_file_manager(self):
        self.file_manager.show(os.getcwd())

    def select_file(self, file_path):
        self.selected_image_path = file_path
        print(f"Selected file: {file_path}")
        self.file_manager.close()
        self.ids.uploaded_image.source = file_path
        self.ids.upload_card.opacity = 1

    def analyze_image(self):
        if not self.selected_image_path:
            print("No image selected!")
            return

        # Run prediction function
        soil_type, ph_value, ph_category, recommended_crops = predict_soil_and_ph(self.selected_image_path)
        
        # Pass results to the results screen
        results_screen = self.manager.get_screen("results")
        results_screen.update_result(soil_type, ph_value, ph_category, recommended_crops)

        # Navigate to results screen
        self.manager.current = "results"

    def open_menu(self):
        """Opens the dropdown menu with navigation options."""
        menu_items = [
            {"viewclass": "OneLineListItem", "text": "Scan","on_release": self.navigate_scan},
            {"viewclass": "OneLineListItem", "text": "Weather","on_release": self.navigate_weather},
        ]
        self.menu = MDDropdownMenu(
            caller=self.ids.open_menu_button,
            items=menu_items,
            width_mult=4
        )
        self.menu.open()

    def navigate_scan(self):
        self.manager.current = "upload"

    def navigate_weather(self):
        self.manager.current = "weather"

class ResultsScreen(Screen):
    def update_result(self, soil_type, ph_value, ph_category, recommended_crops):
        self.ids.result_label.text = f"Soil Type: {soil_type}"
        self.ids.ph_label.text = f"pH Level: {ph_value} ({ph_category})"
        self.ids.crop_label.text = f"Recommended Crops: {', '.join(recommended_crops)}"

    def open_menu(self):
        """Opens the dropdown menu with navigation options."""
        menu_items = [
            {"viewclass": "OneLineListItem", "text": "Scan","on_release": self.navigate_scan},
            {"viewclass": "OneLineListItem", "text": "Weather","on_release": self.navigate_weather},
        ]
        self.menu = MDDropdownMenu(
            caller=self.ids.open_menu_button,
            items=menu_items,
            width_mult=4
        )
        self.menu.open()

    def navigate_scan(self):
        self.manager.current = "upload"

    def navigate_weather(self):
        self.manager.current = "weather"


class WeatherScreen(Screen):
    def fetch_weather(self):
        """Fetches weather details based on the entered city."""
        city = self.ids.city_input.text.strip()

        if city:
            api_key = "24d1f55fb007fb74335ecf1fdb9325bc"  # Replace with your working API key
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

            # Request weather data from API
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                
                # Debug: print raw data to check response from the API
                print(data)

                # Extract weather details
                if "main" in data and "weather" in data:
                    temp = data["main"]["temp"]
                    description = data["weather"][0]["description"]
                    humidity = data["main"]["humidity"]
                    wind_speed = data["wind"]["speed"]

                    # Prepare the weather info text
                    weather_info = f"City: {city}\nTemperature: {temp}°C\nCondition: {description}\nHumidity: {humidity}%\nWind Speed: {wind_speed} m/s"
                    # Update the label with the weather info
                    self.ids.weather_label.text = weather_info
                else:
                    self.ids.weather_label.text = f"Error: Invalid data received for {city}. Please try again."
            else:
                # Error message if API call is unsuccessful
                self.ids.weather_label.text = f"Error: Could not retrieve weather data for {city}. Please try again."
        else:
            # Error message if the city input is empty
            self.ids.weather_label.text = "Please enter a valid city."
    
    def open_menu(self):
        """Opens the dropdown menu with navigation options."""
        menu_items = [
            {"viewclass": "OneLineListItem", "text": "Scan","on_release": self.navigate_scan},
            {"viewclass": "OneLineListItem", "text": "Weather","on_release": self.navigate_weather},
        ]
        self.menu = MDDropdownMenu(
            caller=self.ids.open_menu_button,
            items=menu_items,
            width_mult=4
        )
        self.menu.open()

    def navigate_scan(self):
        self.manager.current = "upload"

    def navigate_weather(self):
        self.manager.current = "weather"

# -------------------------------
# Kivy App Class
# -------------------------------
class SoilScanApp(MDApp):
    def build(self):
        return Builder.load_string(KV)
    def go_back_to_upload(self):
        self.root.current = "upload"

if __name__ == "__main__":
    SoilScanApp().run()
