from flask import Flask
import logging
import sys
from flask import render_template
from main import main, save_document_array
from format_heatmap import format_data


app = Flask(__name__)

app.logger.removeHandler(default_handler := logging.StreamHandler(sys.stdout))
app.logger.addHandler(default_handler)

@app.route("/")
def main_function():
  return render_template("index.html", data=data)

if __name__ == "__main__":
  list_of_mappings = main()
  data = format_data(list_of_mappings)
  app.run()