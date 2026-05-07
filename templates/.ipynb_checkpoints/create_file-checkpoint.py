html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Intrusion Detection System</title>
</head>
<body>

    <h1>Intrusion Detection System</h1>

    <h2>Prediction Result:</h2>

    <p>{{ prediction }}</p>

</body>
</html>
"""

with open("index.html", "w") as file:
    file.write(html_content)

print("index.html created successfully!")