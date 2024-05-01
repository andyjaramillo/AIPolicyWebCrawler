from flask import Flask
from flask import render_template
from main import main

app = Flask(__name__)

@app.route("/")
def main_function():
  pdfs_with_labels,data_metrics = main()
  return render_template("index.html", data=data_metrics)

if __name__ == "__main__":
  app.run()