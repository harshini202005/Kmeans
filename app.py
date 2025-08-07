from flask import Flask, render_template, request
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os

app = Flask(__name__)

# Load model
with open("kmeans_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None

    if request.method == "POST":
        f1 = float(request.form["feature1"])
        f2 = float(request.form["feature2"])
        prediction = model.predict([[f1, f2]])[0]

        # Create and save the updated cluster plot
        df = pd.read_csv("cluster_data.csv")
        plt.figure(figsize=(6, 6))
        for i in df['Cluster'].unique():
            cluster = df[df["Cluster"] == i]
            plt.scatter(cluster["Feature1"], cluster["Feature2"], label=f"Cluster {i}")

        # Plot user input
        plt.scatter(f1, f2, color='black', marker='x', s=100, label="User Input")
        plt.title("KMeans Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()

        os.makedirs("static", exist_ok=True)
        img_path = "static/cluster.png"
        plt.savefig(img_path)
        plt.close()

    return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
