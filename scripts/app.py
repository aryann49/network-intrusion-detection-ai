from flask import Flask, render_template

from scripts.shared_data import alerts

app = Flask(
    __name__,
    template_folder="../templates"
)

@app.route("/")
def home():

    return render_template(
        "index.html",
        alerts=alerts[::-1]
    )

if __name__ == "__main__":
    app.run(debug=True)