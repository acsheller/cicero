from flask import Flask, request, jsonify, render_template_string
import subprocess

app = Flask(__name__)



# HTML template for displaying patterns
patterns_template = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Available Fabric Patterns</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      h1 { color: #333; }
      ul { color: #444; list-style-type: square; }
      .pattern { margin: 10px 0; }
    </style>
  </head>
  <body>
    <h1>Available Fabric Patterns</h1>
    <p>Below is a list of all patterns available in Fabric:</p>
    <ul>
      {% for pattern in patterns %}
        <li class="pattern">{{ pattern }}</li>
      {% endfor %}
    </ul>
  </body>
</html>
"""
# HTML template for the help page
help_html_template = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Fabric REST API Help</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      h1 { color: #333; }
      h2 { color: #666; }
      p, ul { color: #444; }
      ul { list-style-type: square; }
      .endpoint { margin-top: 20px; }
    </style>
  </head>
  <body>
    <h1>Fabric REST API Help</h1>
    <p>This REST API provides access to Fabric commands. Use the endpoints below to interact with Fabric patterns and sessions.</p>
    <div class="endpoint">
      <h2>GET /</h2>
      <p>Displays help information for the REST API in JSON format.</p>
    </div>
    <div class="endpoint">
      <h2>GET /html</h2>
      <p>Displays this help information in a human-readable HTML format.</p>
    </div>
    <div class="endpoint">
      <h2>POST /execute</h2>
      <p>Executes a Fabric pattern with a specified question.</p>
      <p><strong>Request format:</strong></p>
      <ul>
        <li><strong>pattern_name:</strong> Name of the pattern to use.</li>
        <li><strong>question:</strong> The question to ask the pattern.</li>
      </ul>
    </div>
    <div class="endpoint">
      <h2>GET /patterns</h2>
      <p>Lists all available Fabric patterns in JSON format.</p>
    </div>
    <div class="endpoint">
      <h2>GET /patterns/html</h2>
      <p>Lists all available Fabric patterns in a human-readable HTML format.</p>
    </div>
  </body>
</html>
"""

# JSON Help endpoint to display available REST API endpoints and their descriptions
@app.route('/', methods=['GET'])
def api_help():
    help_info = {
        "endpoints": {
            "/": {
                "description": "Displays help information for the REST API in JSON format",
                "method": "GET"
            },
            "/html": {
                "description": "Displays help information for the REST API in human-readable HTML format",
                "method": "GET"
            },
            "/execute": {
                "description": "Executes a Fabric pattern with a specified question.",
                "method": "POST",
                "request_format": {
                    "pattern_name": "Name of the pattern to use",
                    "question": "The question to ask the pattern"
                }
            },
            "/patterns": {
                "description": "Lists all available Fabric patterns in JSON format.",
                "method": "GET"
            },
            "/patterns/html": {
                "description": "Lists all available Fabric patterns in a human-readable HTML format.",
                "method": "GET"
            }
        }
    }
    return jsonify(help_info)

# HTML Help endpoint
@app.route('/html', methods=['GET'])
def api_help_html():
    return render_template_string(help_html_template)

# Endpoint to execute a Fabric pattern
@app.route('/execute', methods=['POST'])
def execute_pattern():
    data = request.json
    pattern_name = data.get("pattern_name")
    question = data.get("question")

    # Run the Fabric command using subprocess
    result = subprocess.run(
        ["fabric", "--pattern", pattern_name, question],
        capture_output=True,
        text=True
    )

    # Return the output of the command as a JSON response
    return jsonify({
        "output": result.stdout,
        "error": result.stderr
    })

# Strictly JSON patterns endpoint for REST
@app.route('/patterns', methods=['GET'])
def list_patterns():
    result = subprocess.run(
        ["fabric", "-l"],
        capture_output=True,
        text=True
    )

    # Parse patterns from command output and return as JSON
    patterns = result.stdout.splitlines()
    return jsonify({"patterns": patterns})

# Optional HTML view for patterns (non-REST endpoint for human-readability)
@app.route('/patterns/html', methods=['GET'])
def list_patterns_html():
    result = subprocess.run(
        ["fabric", "-l"],
        capture_output=True,
        text=True
    )

    # Parse patterns from command output and render as HTML
    patterns = result.stdout.splitlines()
    return render_template_string(patterns_template, patterns=patterns)





if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
